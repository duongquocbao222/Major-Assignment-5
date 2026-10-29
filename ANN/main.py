from __future__ import annotations

import csv
import hashlib
import math
import shutil
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dataset"
PROCESSED = ROOT / "dataset_processed"
COLOR_AUGMENTED = ROOT / "dataset_augmented"
RESULTS = ROOT / "ket_qua"
NUM_CLASSES = 5
NUM_ORIG = 20
ANGLES = [0, 15, -15, 20, -20]
EPOCHS = 200
SEED = 2026
EPS = 1e-10
COLORS = ["#2457a5", "#e27138", "#30946b", "#a552a3", "#c6a12a"]
EXPECTED_DATA_SHA256 = "b14ce7c32e54adcc98c47b33cc5861f256ff78dd04254967b215f47c946f0ab0"

class TerminalAndFile:

    def __init__(self, original, log):
        self.original, self.log = original, log
    def write(self, text):
        self.original.write(text)
        self.log.write(text)
    def flush(self):
        self.original.flush()
        self.log.flush()
    def isatty(self):
        return False
    @property
    def encoding(self):
        return self.original.encoding

def dataset_hash(folder):
    files = sorted(p for p in folder.rglob("*") if p.is_file())
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(folder).as_posix().encode("utf-8"))
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest(), len(files)

def section(title):
    print("\n" + "=" * 96)
    print(title)
    print("=" * 96)

def read_image(path: Path):
    
    buffer = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Khong doc duoc anh: {path}")
    return image

def resize_image(img, max_dim=360):
    h, w = img.shape[:2]
    scale = min(1., max_dim / max(h, w))
    if scale < 1.:
        return cv2.resize(img, (max(1,round(w*scale)), max(1,round(h*scale))),
                          interpolation=cv2.INTER_AREA)
    return img

def keep_largest(mask):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n < 2:
        return np.zeros_like(mask)
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return np.where(labels == largest, 255, 0).astype(np.uint8)


def segment_hand_old(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    skin_y = cv2.inRange(ycrcb, (0, 133, 77), (255, 180, 133))
    skin_h = cv2.inRange(hsv, (0, 20, 35), (27, 255, 255))
    skin_h = cv2.bitwise_or(skin_h, cv2.inRange(hsv, (170, 20, 35), (179, 255, 255)))
    mask = cv2.bitwise_and(skin_y, skin_h)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    return keep_largest(mask)


def segment_hand(bgr):
    img = cv2.GaussianBlur(bgr, (3, 3), 0)
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    S, V = hsv[:, :, 1], hsv[:, :, 2]
    Cr, Cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
    A = lab[:, :, 1]

    skin = ((Cr >= 133) & (Cr <= 185) &
            (Cb >= 75) & (Cb <= 138) &
            (S >= 17) & (V >= 35) & (A >= 129)).astype(np.uint8) * 255
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    skin = keep_largest(skin)
    if np.count_nonzero(skin) < 60:
        print('Canh bao: vung mau da qua it; su dung mask HSV/YCrCb')
        return segment_hand_old(bgr)

    ys, xs = np.where(skin > 0)
    margin = int(0.11 * min(h, w))
    x0, x1 = max(0, int(xs.min()) - margin), min(w, int(xs.max()) + margin + 1)
    y0, y1 = max(0, int(ys.min()) - margin), min(h, int(ys.max()) + margin + 1)

    grab_mask = np.full((h, w), cv2.GC_BGD, np.uint8)
    grab_mask[y0:y1, x0:x1] = cv2.GC_PR_BGD
    expanded = cv2.dilate(skin,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    confident_color = (((Cr >= 143) & (S >= 45)) |
                       ((S >= 95) & (Cr >= 136)))
    confident_skin = cv2.bitwise_and(skin,
                                    confident_color.astype(np.uint8) * 255)
    sure_skin = cv2.erode(confident_skin,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    if np.count_nonzero(sure_skin) < 50:
        sure_skin = skin.copy()

    grab_mask[expanded > 0] = cv2.GC_PR_FGD
    gray_shadow = ((S < 85) & (V < 190) & (Cr <= 140))
    grab_mask[gray_shadow & (sure_skin == 0)] = cv2.GC_BGD
    grab_mask[:2, :] = cv2.GC_BGD
    grab_mask[-2:, :] = cv2.GC_BGD
    grab_mask[:, :2] = cv2.GC_BGD
    grab_mask[:, -2:] = cv2.GC_BGD
    grab_mask[sure_skin > 0] = cv2.GC_FGD

    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(img, grab_mask, None, bg_model, fg_model,
                    3, cv2.GC_INIT_WITH_MASK)
        mask = np.isin(grab_mask, [cv2.GC_FGD, cv2.GC_PR_FGD]).astype(np.uint8) * 255
    except cv2.error:
        print('Canh bao: GrabCut khong hoi tu; su dung mask HSV/YCrCb')
        return segment_hand_old(bgr)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    mask = keep_largest(mask)

    inverse = cv2.bitwise_not(mask)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inverse, connectivity=8)
    hole_limit = max(6, int(0.00045 * h * w))
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        if area <= hole_limit and x > 0 and y > 0 and x + ww < w and y + hh < h:
            mask[labels == i] = 255

    ratio = np.count_nonzero(mask) / mask.size
    if not 0.008 < ratio < 0.85:
        print('Canh bao: ti le mask bat thuong; su dung HSV/YCrCb')
        return segment_hand_old(bgr)
    return mask


def seg_overlay(image, old_mask, new_mask):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    out = rgb.copy()
    foreground = new_mask > 0
    removed = (old_mask > 0) & ~foreground
    out[foreground] = 0.55 * rgb[foreground] + 0.45 * np.array([30, 205, 75])
    out[removed] = 0.35 * rgb[removed] + 0.65 * np.array([230, 35, 40])
    return np.uint8(np.clip(out, 0, 255))


def centroid_from_mask(mask):
    moments = cv2.moments(mask,binaryImage=True)
    m00 = moments["m00"]
    if m00 <= 0:
        raise ValueError("Khong co pixel tay trong mask")
    return float(moments["m10"]/m00),float(moments["m01"]/m00),moments

def rotate_mask(mask, angle, cx, cy):

    h,w = mask.shape
    
    side = int(math.ceil(math.hypot(h,w)))+16
    pad_l = (side-w)//2
    pad_t = (side-h)//2
    large = cv2.copyMakeBorder(mask,pad_t,side-h-pad_t,pad_l,side-w-pad_l,
                               cv2.BORDER_CONSTANT,value=0)
    mat = cv2.getRotationMatrix2D((cx+pad_l,cy+pad_t),angle,1.0)
    out = cv2.warpAffine(large,mat,(side,side),flags=cv2.INTER_NEAREST,
                         borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    return np.where(out>127,255,0).astype(np.uint8)

def rotate_color_native(image, angle, cx, cy):
    h, w = image.shape[:2]
    side = int(math.ceil(math.hypot(h, w))) + 16
    left = (side - w) // 2
    top = (side - h) // 2
    canvas = cv2.copyMakeBorder(image, top, side - h - top,
                                left, side - w - left,
                                cv2.BORDER_CONSTANT, value=(255, 255, 255))
    transform = cv2.getRotationMatrix2D((cx + left, cy + top), angle, 1.0)
    return cv2.warpAffine(canvas, transform, (side, side),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(255, 255, 255))

def hu_features(mask):
    mom = cv2.moments(mask,binaryImage=True)
    hu = cv2.HuMoments(mom).reshape(7).astype(np.float64)
    hu_log = -np.sign(hu)*np.log10(np.abs(hu)+EPS)
    if not np.all(np.isfinite(hu_log)):
        raise ValueError("Hu Moments bi NaN/Inf")
    return hu, hu_log, mom

def image_fingerprint(binary_mask):

    resized=cv2.resize(binary_mask,(32,32),interpolation=cv2.INTER_AREA)
    dct=cv2.dct(np.float32(resized))
    values=dct[:8,:8].ravel()
    return values > np.median(values[1:])

def split_originals_by_similarity(paths, original_masks, seed):

    n=len(paths)
    fps=[]
    for mask in original_masks:
        cx,cy,_=centroid_from_mask(mask)
        fps.append(image_fingerprint(rotate_mask(mask,0,cx,cy)))
    parent=list(range(n))
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]
            i=parent[i]
        return i
    for i in range(n):
        for j in range(i):
            if int(np.count_nonzero(fps[i]!=fps[j]))<=4:
                parent[root(i)]=root(j)
    groups_by_root={}
    for i in range(n):
        groups_by_root.setdefault(root(i),[]).append(i)
    groups=list(groups_by_root.values())
    local_rng=np.random.default_rng(seed)
    groups=[groups[i] for i in local_rng.permutation(len(groups))]
    sizes=tuple(len(g) for g in groups)
    
    @lru_cache(None)
    def assign(k, ntrain, nval):
        if k==len(sizes):
            return () if (ntrain,nval)==(14,3) else None
        ntest=sum(sizes[:k])-ntrain-nval
        for split_i in (0,1,2):
            if split_i==0 and ntrain+sizes[k]<=14:
                rest=assign(k+1,ntrain+sizes[k],nval)
            elif split_i==1 and nval+sizes[k]<=3:
                rest=assign(k+1,ntrain,nval+sizes[k])
            elif split_i==2 and ntest+sizes[k]<=3:
                rest=assign(k+1,ntrain,nval)
            else:
                continue
            if rest is not None:
                return (split_i,)+rest
        return None
    plan=assign(0,0,0)
    if plan is None:
        raise RuntimeError('Khong the chia 14/3/3 khi giu cac cap anh gan trung. '
                           'Can thu thap them anh doc lap.')
    split={}
    for group,split_i in zip(groups,plan):
        for idx in group:
            split[idx]=('Train','Validation','Test')[split_i]
    assert Counter(split.values())=={'Train':14,'Validation':3,'Test':3}
    print('Phat hien',len(groups),'nhom anh doc lap, kich thuoc:',sorted(sizes,reverse=True))
    print('Kiem tra nhom anh gan trung chi nam trong mot tap: DAT')
    return split

def save_csv(path, rows, columns):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f,fieldnames=columns,extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

def prep_dataset(rng):
    section("BUOC 1 - KIEM TRA DU LIEU, CHIA ANH GOC, XOAY VA GAN ID")
    print("cx = m10 / m00 ; cy = m01 / m00")
    print("Goc theta:",ANGLES,"=> 20 anh goc * 5 goc = 100 anh/class")
    all_records = []
    preview = []
    split_target = {'Train':0,'Validation':0,'Test':0}
    for ci in range(1,NUM_CLASSES+1):
        folder = DATA/f'class_{ci}'
        if not folder.is_dir():
            raise FileNotFoundError(f"Can {folder}; giai nen ZIP dataset vao thu muc nay")
        paths = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in
                       ('.jpg','.jpeg','.png','.bmp','.webp'))
        if len(paths) != NUM_ORIG:
            raise ValueError(f"class_{ci} co {len(paths)} anh; de yeu cau DUNG {NUM_ORIG}")

        original_images=[resize_image(read_image(p)) for p in paths]
        baseline_masks=[segment_hand_old(img) for img in original_images]
        original_masks=[segment_hand(img) for img in original_images]
        split_map=split_originals_by_similarity(paths,baseline_masks,SEED+ci*17)
        samples = []
        for oi,path in enumerate(paths):
            bgr = original_images[oi]
            gray = cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
            mask = original_masks[oi]
            cx,cy,mom = centroid_from_mask(mask)
            if ci == 1 and oi == 0:
                print(f"\nClass {ci} - thay so centroid, anh dau: {path.name}")
                print(f"m00={mom['m00']:.0f}; m10={mom['m10']:.4f}; m01={mom['m01']:.4f}")
                print(f"cx={mom['m10']:.4f}/{mom['m00']:.0f}={cx:.6f}")
                print(f"cy={mom['m01']:.4f}/{mom['m00']:.0f}={cy:.6f}")
                print(f"Vi du +15 do: cos15={math.cos(math.radians(15)):.6f}; "
                      f"sin15={math.sin(math.radians(15)):.6f}")
                ex_x,ex_y = cx+20,cy+10
                tx = cx + 20*math.cos(math.radians(15)) + 10*math.sin(math.radians(15))
                ty = cy - 20*math.sin(math.radians(15)) + 10*math.cos(math.radians(15))
                print(f"Diem P=({ex_x:.6f},{ex_y:.6f}), P sau xoay +15="
                      f"({cx:.5f}+20*cos15+10*sin15,"
                      f"{cy:.5f}-20*sin15+10*cos15)=({tx:.6f},{ty:.6f})")
                
            if oi == 0:
                preview.append((ci,bgr.copy(),gray.copy(),baseline_masks[oi].copy(),
                                mask.copy(),rotate_mask(mask,15,cx,cy)))
            for angle in ANGLES:
                rotated = rotate_mask(mask,angle,cx,cy)
                hu,hu_log,_ = hu_features(rotated)
                samples.append({'class':ci,'source':path.name,'source_no':oi+1,
                                'split':split_map[oi],'angle':angle,'mask':rotated,
                                'centroid_x':cx,'centroid_y':cy,'hu':hu,'hu_log':hu_log})
        
        order = rng.permutation(len(samples))
        class_records=[]
        for local_index,sample_idx in enumerate(order):
            record = samples[int(sample_idx)]
            uid = (ci-1)*100+local_index+1
            dest_folder = PROCESSED/f'class_{ci}'
            dest_folder.mkdir(exist_ok=True,parents=True)
            out_name = f"ID_{uid:03d}_{record['split']}_goc_{record['angle']:+03d}.png"
            out_path = dest_folder/out_name
            if not cv2.imwrite(str(out_path),record.pop('mask')):
                raise OSError(f"Khong ghi duoc: {out_path}")
            record['id']=uid
            record['processed_image']=str(out_path.relative_to(ROOT)).replace('\\','/')
            all_records.append(record)
            class_records.append(record)
            split_target[record['split']]+=1
        for oi, source_path in enumerate(paths):
            group = [rec for rec in class_records if rec['source_no'] == oi + 1]
            if len(group) != len(ANGLES):
                raise RuntimeError(f'Sai so anh augmentation: {source_path.name}')
            color_original = read_image(source_path)
            src_h, src_w = original_images[oi].shape[:2]
            full_h, full_w = color_original.shape[:2]
            color_folder = COLOR_AUGMENTED / f'class_{ci}'
            color_folder.mkdir(parents=True, exist_ok=True)
            for rec in group:
                center_x = rec['centroid_x'] * full_w / src_w
                center_y = rec['centroid_y'] * full_h / src_h
                filename = Path(rec['processed_image']).stem + '.jpg'
                target = color_folder / filename
                if target.exists() and '--reuse-colors' in sys.argv:
                    pass
                elif rec['angle'] == 0:
                    shutil.copyfile(source_path, target)
                else:
                    color_rotated = rotate_color_native(color_original, rec['angle'],
                                                       center_x, center_y)
                    if not cv2.imwrite(str(target), color_rotated,
                                       [cv2.IMWRITE_JPEG_QUALITY, 92]):
                        raise OSError(f'Khong luu duoc anh mau: {target}')
                rec['augmented_color_image'] = target.relative_to(ROOT).as_posix()
        print(f'Class {ci}: da luu 100 anh MAU xoay rieng (khong nhi phan hoa)')
        counts=Counter(rec['split'] for rec in class_records)
        assert len(class_records)==100 and counts=={'Train':70,'Validation':15,'Test':15}
        print(f"Class {ci}: 20 goc -> 100 anh nhị phân; IDs {(ci-1)*100+1:03d}..{ci*100:03d}; "
              f"Train={counts['Train']}, Val={counts['Validation']}, Test={counts['Test']}")
    color_count = sum(1 for file in COLOR_AUGMENTED.rglob('ID_*.jpg'))
    binary_count = sum(1 for file in PROCESSED.rglob('ID_*.png'))
    print(f'TONG ANH MAU AUGMENTATION: {color_count} | TONG ANH NHI PHAN: {binary_count}')
    assert color_count == 500 and binary_count == 500
    print("TONG KET:",split_target,'; N =',len(all_records))
    assert len(all_records)==500 and len({x['id'] for x in all_records})==500
    assert split_target=={'Train':350,'Validation':75,'Test':75}
    
    for ci in range(1,6):
        sr={}
        for r in all_records:
            if r['class']==ci:
                sr.setdefault(r['source'],set()).add(r['split'])
        assert all(len(v)==1 for v in sr.values())
    print("KIEM TRA: moi anh goc va 5 phien ban xoay cung trong 1 tap: DAT")
    return sorted(all_records,key=lambda r:r['id']),preview

def fig_setup():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,
                         'axes.labelsize':10,'figure.dpi':110,'savefig.dpi':220,
                         'axes.grid':False})

def export_previews(preview):
    section("BUOC 2 - SO SANH ANH GOC, MASK CU, MASK MOI, VUNG LOAI BO")
    print('Mau da HSV/YCrCb/Lab + GrabCut 3 lan + loai bong xam + giu vung lien thong lon nhat')
    print('Anh nhi phan: tay = 255 (trang), nen va bong = 0 (den)')
    fig, axes = plt.subplots(5, 6, figsize=(17, 16))
    titles = ['Anh goc', 'Anh xam', 'Mask cu', 'Mask moi',
              'Xanh: tay | Do: bi loai', 'Xoay +15 do']
    for row, (ci, bgr, gray, old, mask, rotated) in enumerate(preview):
        imgs = [cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), gray, old,
                mask, seg_overlay(bgr, old, mask), rotated]
        for col, im in enumerate(imgs):
            axes[row, col].imshow(im, cmap='gray' if im.ndim == 2 else None,
                                  vmin=0 if im.ndim == 2 else None,
                                  vmax=255 if im.ndim == 2 else None)
            axes[row, col].axis('off')
            if row == 0:
                axes[row, col].set_title(titles[col], fontsize=10)
            if col == 0:
                axes[row, col].set_ylabel(f'Class {ci}', fontsize=12)
        changed = 100 * np.mean((old > 0) != (mask > 0))
        print(f'Class {ci}: cu {100*np.mean(old > 0):.1f}% trang, '
              f'moi {100*np.mean(mask > 0):.1f}% trang, '
              f'{changed:.1f}% pixel mask thay doi')
    fig.tight_layout()
    fig.savefig(RESULTS / '01_tien_xu_ly_va_xoay.png', dpi=230, bbox_inches='tight')
    plt.close(fig)

    fig, axes = plt.subplots(15, 4, figsize=(11, 37))
    for ci in range(1, 6):
        folder = DATA / f'class_{ci}'
        files = sorted(p for p in folder.iterdir() if p.suffix.lower() in
                       ('.jpg', '.jpeg', '.png', '.bmp', '.webp'))
        for j, index in enumerate((0, 5, 10)):
            original = resize_image(read_image(files[index]))
            old = segment_hand_old(original)
            new = segment_hand(original)
            row = (ci - 1) * 3 + j
            views = [cv2.cvtColor(original, cv2.COLOR_BGR2RGB), old, new,
                     seg_overlay(original, old, new)]
            for col, im in enumerate(views):
                ax = axes[row, col]
                ax.imshow(im, cmap='gray' if im.ndim == 2 else None,
                          vmin=0 if im.ndim == 2 else None,
                          vmax=255 if im.ndim == 2 else None)
                ax.axis('off')
                if row == 0:
                    ax.set_title(['Goc', 'Mask cu', 'Mask GrabCut', 'So sanh'][col])
                if col == 0:
                    ax.set_ylabel(f'C{ci} - anh {index+1}', fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS / '09_so_sanh_mask_cu_moi_15_anh.png', dpi=200,
                bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    xs = np.arange(5)
    ax.bar(xs, [70] * 5, color='#487ab3', label='Train')
    ax.bar(xs, [15] * 5, bottom=[70] * 5, color='#d68a48', label='Validation')
    ax.bar(xs, [15] * 5, bottom=[85] * 5, color='#60a78d', label='Test')
    ax.set_xticks(xs, [f'Class {i}' for i in range(1, 6)])
    ax.set(ylabel='So anh', title='Train / Validation / Test')
    ax.set_ylim(0, 110)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / '20_phan_chia_5_class.png')
    plt.close(fig)
    print('Da luu 01_tien_xu_ly_va_xoay.png va 09_so_sanh_mask_cu_moi_15_anh.png')

def export_five_angles(records):

    fig,axes=plt.subplots(5,5,figsize=(13,13))
    for ci in range(1,6):
        same=[r for r in records if r['class']==ci and r['source_no']==1]
        by_angle={r['angle']:r for r in same}
        for col,angle in enumerate(ANGLES):
            r=by_angle[angle]
            im=cv2.imread(str(ROOT/r['processed_image']),cv2.IMREAD_GRAYSCALE)
            ax=axes[ci-1,col]
            ax.imshow(im,cmap='gray',vmin=0,vmax=255)
            ax.axis('off')
            if ci==1:ax.set_title(f'Goc {angle:+d}°',fontsize=12)
            ax.text(.03,.02,f'Class {ci} | ID {r["id"]:03d}',transform=ax.transAxes,
                    color='white',fontsize=8,va='bottom',
                    bbox={'facecolor':'black','alpha':.6,'pad':2})
    fig.suptitle('Data augmentation: 1 anh goc -> 5 goc xoay quanh centroid',fontsize=15)
    fig.tight_layout(rect=[0,0,1,.975])
    fig.savefig(RESULTS/'00_nam_goc_xoay_theo_centroid.png');plt.close(fig)
    print('Da luu ket_qua/00_nam_goc_xoay_theo_centroid.png')

def export_color_angles(records):
    fig, axes = plt.subplots(5, 5, figsize=(14, 13))
    for ci in range(1, 6):
        same = [r for r in records if r['class'] == ci and r['source_no'] == 1]
        by_angle = {r['angle']: r for r in same}
        for col, angle in enumerate(ANGLES):
            rec = by_angle[angle]
            image = read_image(ROOT / rec['augmented_color_image'])
            axes[ci-1, col].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            axes[ci-1, col].axis('off')
            if ci == 1:
                axes[ci-1, col].set_title(f'Goc {angle:+d}°')
            if col == 0:
                axes[ci-1, col].set_ylabel(f'Class {ci}', fontsize=12)
            axes[ci-1, col].text(.02, .02, f'ID {rec["id"]:03d}',
                                    transform=axes[ci-1, col].transAxes,
                                    color='black', fontsize=8,
                                    bbox={'facecolor':'white','alpha':.75,'pad':2})
    fig.suptitle('Augmentation anh mau goc, chua nhi phan hoa', fontsize=15)
    fig.tight_layout(rect=[0,0,1,.975])
    fig.savefig(RESULTS / '10_anh_mau_500_augmentation.png', dpi=220)
    plt.close(fig)
    print('Da luu ket_qua/10_anh_mau_500_augmentation.png')

def arrays_from_records(records):
    Xlog=np.stack([r['hu_log'] for r in records])
    y=np.array([r['class']-1 for r in records],np.int64)
    splits=np.array([r['split'] for r in records])
    train=splits=='Train';val=splits=='Validation';test=splits=='Test'
    scaler=MinMaxScaler(feature_range=(0,1),clip=True)
    X=np.empty_like(Xlog)
    X[train]=scaler.fit_transform(Xlog[train])  
    X[val]=scaler.transform(Xlog[val])
    X[test]=scaler.transform(Xlog[test])
    assert np.all((0<=X)&(X<=1))

    section("BUOC 3 - TRICH HU, LOGARIT VA CHUAN HOA")
    print("Hu1 = eta20 + eta02; Hu2 = (eta20-eta02)^2 + 4*eta11^2")
    print("Hu3 = (eta30-3*eta12)^2 + (3*eta21-eta03)^2")
    print("h_log = -sign(h)*log10(abs(h)+1e-10)")
    print("MinMax = (h_log - min_train)/(max_train-min_train)")
    r=next(r for r in records if r['class']==1 and r['split']=='Train')
    idx=records.index(r)
    m=cv2.moments(cv2.imread(str(ROOT/r['processed_image']),0),binaryImage=True)
    nu20,nu02,nu11=m['nu20'],m['nu02'],m['nu11']
    print(f"Vi du CLASS 1, ID {r['id']:03d}, tap TRAIN:")
    print(f"Hu1 = {nu20:.8f} + {nu02:.8f} = {r['hu'][0]:.8f}")
    print(f"Hu2 = ({nu20:.8f}-{nu02:.8f})^2 + 4*({nu11:.8f})^2 = {r['hu'][1]:.8f}")
    for j in range(3):
        h=float(r['hu'][j]);z=float(r['hu_log'][j])
        mn=float(scaler.data_min_[j]);mx=float(scaler.data_max_[j]);v=float(X[idx,j])
        print(f"Hu{j+1}: log=-sign({h:+.5e})*log10(abs({h:+.5e})+1e-10)={z:.6f}")
        print(f"     MinMax=({z:.6f}-{mn:.6f})/({mx:.6f}-{mn:.6f})={v:.6f}")
    print("Train: 350x7 | Validation: 75x7 | Test: 75x7 | fit scaler: Train")
    meta=[]
    for i,r in enumerate(records):
        row={'ID':r['id'],'Class':r['class'],'Tap':r['split'],'Anh_goc':r['source'],
             'So_anh_goc':r['source_no'],'Goc_xoay':r['angle'],
             'Anh_sau_xu_ly':r['processed_image'], 'Anh_mau_augmented':r['augmented_color_image'],'Centroid_X':f"{r['centroid_x']:.6f}",
             'Centroid_Y':f"{r['centroid_y']:.6f}"}
        for j in range(7):
            row[f'Hu{j+1}_raw']=f"{r['hu'][j]:.12e}"
            row[f'Hu{j+1}_log']=f"{r['hu_log'][j]:.12f}"
            row[f'Hu{j+1}_scaled']=f"{X[i,j]:.12f}"
        meta.append(row)
    save_csv(RESULTS/'du_lieu_500_ID_Hu.csv',meta,list(meta[0]))
    print("Da luu ket_qua/du_lieu_500_ID_Hu.csv")
    return X,y,train,val,test

def export_scatter(X,y,train):
    section("BUOC 3B - TRUC QUAN HOA HU1-HU2 VA HU1-HU2-HU3 (TRAIN)")
    
    fig,ax=plt.subplots(figsize=(9,7))
    for c in range(5):
        mask=(y==c)&train
        ax.scatter(X[mask,0],X[mask,1],s=36,alpha=.74,color=COLORS[c],
                   edgecolors='white',linewidths=.25,label=f'Class {c+1}')
    ax.set(xlabel='Hu1 (MinMax)',ylabel='Hu2 (MinMax)',title='Scatter 2D: Hu1, Hu2 - Training Set')
    ax.grid(alpha=.22);ax.legend();fig.tight_layout()
    fig.savefig(RESULTS/'02_scatter_2D_Hu1_Hu2.png');plt.close(fig)
    fig=plt.figure(figsize=(10,8));ax=fig.add_subplot(111,projection='3d')
    for c in range(5):
        mask=(y==c)&train
        ax.scatter(X[mask,0],X[mask,1],X[mask,2],s=30,alpha=.8,
                   color=COLORS[c],label=f'Class {c+1}')
    ax.set(xlabel='Hu1 MinMax',ylabel='Hu2 MinMax',zlabel='Hu3 MinMax')
    ax.set_title('Scatter 3D: Hu1, Hu2, Hu3 - Training Set');ax.legend()
    fig.tight_layout();fig.savefig(RESULTS/'03_scatter_3D_Hu1_Hu2_Hu3.png');plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(14,9))
    pairs=[(0,1),(0,2),(1,2),(0,3),(1,3),(2,3)]
    for ax,(a,b) in zip(axes.ravel(),pairs):
        for c in range(5):
            good=(y==c)&train
            ax.scatter(X[good,a],X[good,b],s=16,alpha=.6,color=COLORS[c],label=f'C{c+1}')
        ax.set(xlabel=f'Hu{a+1}',ylabel=f'Hu{b+1}')
        ax.grid(alpha=.15)
    axes[0,0].legend(ncol=5,fontsize=8)
    fig.suptitle('Six feature-pair views on training data')
    fig.tight_layout()
    fig.savefig(RESULTS/'11_sau_cap_dac_trung_Hu.png')
    plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(16,8))
    for j,ax in enumerate(axes.ravel()):
        if j==7:
            ax.axis('off')
            continue
        values=[X[train&(y==c),j] for c in range(5)]
        bp=ax.boxplot(values,patch_artist=True,showfliers=False)
        for patch,color in zip(bp['boxes'],COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(.55)
        ax.set_title(f'Hu{j+1} - Training')
        ax.set_xticks(range(1,6),[f'C{i}' for i in range(1,6)])
        ax.grid(axis='y',alpha=.2)
    fig.tight_layout()
    fig.savefig(RESULTS/'12_phan_bo_7_dac_trung_Hu.png')
    plt.close(fig)
    print("Da luu scatter 2D va 3D trong ket_qua/")
    print("NHAN XET: Neu mau cac class de len nhau tren scatter, 2/3 Hu dau chua tach lop tot.")

class SimpleANN:
    def __init__(self, hidden, seed=SEED):
        rng=np.random.default_rng(seed)
        lim1=np.sqrt(6/(7+hidden));lim2=np.sqrt(6/(hidden+5))
        self.W1=rng.uniform(-lim1,lim1,(7,hidden))
        self.b1=np.zeros((1,hidden),dtype=np.float64)
        self.W2=rng.uniform(-lim2,lim2,(hidden,5))
        self.b2=np.zeros((1,5),dtype=np.float64)
    @staticmethod
    def sigmoid(z):
        
        return 1/(1+np.exp(-np.clip(z,-50,50)))
    def forward(self,X):
        z1=X@self.W1+self.b1
        a1=self.sigmoid(z1)
        z2=a1@self.W2+self.b2
        z2=z2-np.max(z2,axis=1,keepdims=True)
        e=np.exp(z2)
        p=e/np.sum(e,axis=1,keepdims=True)
        return z1,a1,z2,p
    @staticmethod
    def cce(p,Y):
        return float(-np.sum(Y*np.log(np.clip(p,1e-12,1.0)))/len(Y))
    def gradients(self,X,Y):
        z1,a1,z2,p=self.forward(X)
        n=len(X)
        dz2=(p-Y)/n
        dw2=a1.T@dz2;db2=np.sum(dz2,axis=0,keepdims=True)
        dz1=(dz2@self.W2.T)*a1*(1-a1)
        dw1=X.T@dz1;db1=np.sum(dz1,axis=0,keepdims=True)
        return (dw1,db1,dw2,db2),(z1,a1,z2,p,dz2,dz1)
    def update(self,grads,lr):
        dw1,db1,dw2,db2=grads
        self.W1-=lr*dw1;self.b1-=lr*db1
        self.W2-=lr*dw2;self.b2-=lr*db2
    def snapshot(self):
        return tuple(np.copy(p) for p in (self.W1,self.b1,self.W2,self.b2))
    def restore(self,snap):
        self.W1,self.b1,self.W2,self.b2=tuple(np.copy(p) for p in snap)

def one_hot(y):
    return np.eye(5,dtype=np.float64)[y]

def show_manual_backprop(model, X, Y, lr):
    
    (dw1,db1,dw2,db2),(z1,a1,z2,p,dz2,dz1)=model.gradients(X,Y)
    x=X[0];n=len(X)
    print("\n******** VI DU THE SO LAN TRUYEN THUAN + LAN TRUYEN NGUOC ********")
    print(f"Batch N={n}, X[0]={np.array2string(x,precision=7)}")
    print(f"Nhan class={int(np.argmax(Y[0]))+1}; one-hot={Y[0].astype(int)}")
    print("Z1 = X @ W1 + b1, A1 = 1/(1+exp(-Z1))")
    print("Z2 = A1 @ W2 + b2, Pk = exp(Z2k)/sum(exp(Z2j))")
    expr=' + '.join(f"({x[j]:.6f})*({model.W1[j,0]:+.6f})" for j in range(7))
    print(f"Z1[0,0] = {expr} + ({model.b1[0,0]:+.6f}) = {z1[0,0]:+.9f}")
    print(f"A1[0,0] = 1/(1+exp(-({z1[0,0]:+.9f}))) = {a1[0,0]:.9f}")
    expr2=' + '.join(f"({a1[0,j]:.5f})*({model.W2[j,0]:+.5f})" for j in range(model.W2.shape[0]))
    z2_raw=a1[0]@model.W2+model.b2[0]
    print(f"Z2[0,0] = {expr2} + ({model.b2[0,0]:+.5f}) = {z2_raw[0]:+.9f}")
    print(f"P[0] softmax={np.array2string(p[0],precision=9)}")
    k=int(np.argmax(Y[0]));print(f"CCE mau 0=-ln(P_dung)=-ln({p[0,k]:.9f})={-np.log(p[0,k]):.9f}")
    print(f"CCE ca batch = -1/{n} * SUM_i SUM_k (y_ik ln p_ik) = {model.cce(p,Y):.9f}")
    print("dZ2 = (P-Y)/N")
    print(f"dZ2[0,0]=({p[0,0]:.9f}-{Y[0,0]:.0f})/{n}={dz2[0,0]:+.12f}")
    print("dW2=A1^T @ dZ2; db2=SUM(dZ2)")
    print(f"dW2[0,0]=SUM_i A1[i,0]*dZ2[i,0]={dw2[0,0]:+.12f}")
    print("dZ1=(dZ2 @ W2^T)*A1*(1-A1); dW1=X^T @ dZ1")
    print(f"dZ1[0,0]={dz1[0,0]:+.12f}, dW1[0,0]={dw1[0,0]:+.12f}")
    print(f"W1_old[0,0]={model.W1[0,0]:+.12f}")
    print(f"W1_new[0,0] = {model.W1[0,0]:+.12f} - ({lr})*({dw1[0,0]:+.12f}) "
          f"= {model.W1[0,0]-lr*dw1[0,0]:+.12f}")
    print(f"W2_new[0,0] = {model.W2[0,0]:+.12f} - ({lr})*({dw2[0,0]:+.12f}) "
          f"= {model.W2[0,0]-lr*dw2[0,0]:+.12f}")
    print("Toan bo 350 mau dong gop vao gradient, khong cap nhat theo tung mau.")
    print("***************************************************************\n")

def train_models(X,y,train,val):
    section("BUOC 4 - TRAIN 8 CAU HINH: 7-H-5; H=7,8,9,10; LR=0.01,0.1")
    Xt=X[train];yt=y[train]; Xv=X[val];yv=y[val]
    Yt=one_hot(yt);Yv=one_hot(yv)
    print("Input=7, Hidden=7/8/9/10 (Sigmoid), Output=5 (Softmax), Loss=CCE")
    print("Batch GD: Batch size N=350, epoch=200, W duoc khoi tao ngau nhien Glorot.")
    print("Val chi forward+tinh CCE, TUYET DOI KHONG dung de cap nhat trong so")
    print("Validation LOSS nho nhat -> chon checkpoint, KHONG dung test de chon model")
    print("Moi cau hinh: epoch 1, 25, 50, 100, 150, 200.")
    histories=[]
    best_global=None
    for hidden in (7,8,9,10):
        for lr in (0.01,0.1):
            model=SimpleANN(hidden,seed=SEED+hidden*47)
            if hidden==7 and lr==0.01:
                show_manual_backprop(model,Xt,Yt,lr)
            history={'hidden':hidden,'lr':lr,'train_loss':[], 'val_loss':[],
                     'train_acc':[],'val_acc':[], 'best_epoch':None,'best_val_loss':float('inf')}
            best_weights=None
            print(f"\n>>> ANN 7-{hidden}-5; learning rate eta={lr}; epochs={EPOCHS}")
            for epoch in range(1,EPOCHS+1):
                grads,_=model.gradients(Xt,Yt)
                model.update(grads,lr)
                Ptrain=model.forward(Xt)[3]
                Pval=model.forward(Xv)[3]
                tr_loss=model.cce(Ptrain,Yt)
                va_loss=model.cce(Pval,Yv)
                tr_acc=float(np.mean(np.argmax(Ptrain,axis=1)==yt))
                va_acc=float(np.mean(np.argmax(Pval,axis=1)==yv))
                history['train_loss'].append(tr_loss)
                history['val_loss'].append(va_loss)
                history['train_acc'].append(tr_acc)
                history['val_acc'].append(va_acc)
                if va_loss < history['best_val_loss']-1e-12:
                    history['best_val_loss']=va_loss
                    history['best_epoch']=epoch
                    best_weights=model.snapshot()
                if epoch in (1, 25, 50, 100, 150, 200):
                    print(f"  Epoch {epoch:03d}: Train CCE={tr_loss:.5f}, "
                          f"Val CCE={va_loss:.5f}, Train Acc={tr_acc*100:.2f}%, "
                          f"Val Acc={va_acc*100:.2f}%")
            model.restore(best_weights)
            history['best_weights']=best_weights
            history['best_model']=model
            histories.append(history)
            print(f"KET THUC 7-{hidden}-5 eta={lr}: epoch tot nhat={history['best_epoch']}, "
                  f"Val CCE={history['best_val_loss']:.9f}")
            if best_global is None or history['best_val_loss']<best_global['best_val_loss']:
                best_global=history
    print("\nCHON THEO VALIDATION LOSS, KHONG THEO TEST:")
    print(f"ANN 7-{best_global['hidden']}-5; eta={best_global['lr']}; "
          f"epoch={best_global['best_epoch']}; Val CCE={best_global['best_val_loss']:.9f}")
    rows=[]
    for h in histories:
        e=h['best_epoch']-1
        rows.append({'Kien_truc':f"7-{h['hidden']}-5",'Learning_Rate':h['lr'],
                     'Best_Epoch':h['best_epoch'],'Training_CCE_At_Best':f"{h['train_loss'][e]:.9f}",
                     'Validation_CCE_Best':f"{h['best_val_loss']:.9f}",
                     'Training_Accuracy_At_Best':f"{100*h['train_acc'][e]:.4f}",
                     'Validation_Accuracy_At_Best':f"{100*h['val_acc'][e]:.4f}"})
    save_csv(RESULTS/'so_sanh_8_mo_hinh.csv',rows,list(rows[0]))
    return histories,best_global

def export_losses(histories,best):
    section("BUOC 4B - VE DO THI HUAN LUYEN VA SO SANH")
    fig,axes=plt.subplots(4,2,figsize=(14,17),sharex=True)
    for h,ax in zip(histories,axes.ravel()):
        t=np.arange(1,EPOCHS+1)
        ax.plot(t,h['train_loss'],color='#2469b4',lw=1.8,label='Training CCE')
        ax.plot(t,h['val_loss'],color='#dd6434',lw=1.8,label='Validation CCE')
        ax.axvline(h['best_epoch'],ls='--',color='gray',lw=1)
        ax.set_title(f"7-{h['hidden']}-5 | eta={h['lr']} | best epoch={h['best_epoch']}")
        ax.set_ylabel('CCE');ax.grid(alpha=.24)
        ax.legend(fontsize=8)
    for ax in axes[-1,:]:ax.set_xlabel('Epoch')
    fig.suptitle('8 THU NGHIEM ANN - TRAINING vs VALIDATION LOSS',fontsize=16)
    fig.tight_layout(rect=[0,0,1,.98])
    fig.savefig(RESULTS/'04_so_sanh_8_do_thi_loss.png');plt.close(fig)
    
    fig,axes=plt.subplots(2,2,figsize=(13,10))
    for hidden,ax in zip((7,8,9,10),axes.ravel()):
        for h in histories:
            if h['hidden']==hidden:
                ax.plot(np.arange(1,EPOCHS+1),h['val_loss'],lw=2,
                        label=f"LR={h['lr']}")
        ax.set(title=f'Hidden={hidden}: Validation Loss / hai learning rates',xlabel='Epoch',ylabel='Val CCE')
        ax.grid(alpha=.2);ax.legend()
    fig.tight_layout();fig.savefig(RESULTS/'05_so_sanh_learning_rate.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,6));t=np.arange(1,EPOCHS+1)
    ax.plot(t,best['train_loss'],lw=2.3,color='#2668b0',label='Training Loss')
    ax.plot(t,best['val_loss'],lw=2.3,color='#df733f',label='Validation Loss')
    ax.axvline(best['best_epoch'],color='gray',ls='--',label=f"Best epoch={best['best_epoch']}")
    ax.set(title=f"BEST ANN 7-{best['hidden']}-5 - eta={best['lr']}",xlabel='Epoch',ylabel='CCE')
    ax.grid(alpha=.25);ax.legend();fig.tight_layout()
    fig.savefig(RESULTS/'06_loss_mo_hinh_tot_nhat.png');plt.close(fig)
    fig,axes=plt.subplots(4,2,figsize=(14,16),sharex=True,sharey=True)
    for h,ax in zip(histories,axes.ravel()):
        t=np.arange(1,EPOCHS+1)
        ax.plot(t,100*np.array(h['train_acc']),lw=1.9,label='Train',color='#2457a5')
        ax.plot(t,100*np.array(h['val_acc']),lw=1.9,label='Validation',color='#df733f')
        ax.axvline(h['best_epoch'],color='gray',ls='--',lw=1)
        ax.set(title=f"ANN 7-{h['hidden']}-5 | LR={h['lr']}",ylabel='Accuracy (%)')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    axes[-1,0].set_xlabel('Epoch');axes[-1,1].set_xlabel('Epoch')
    fig.tight_layout()
    fig.savefig(RESULTS/'13_tam_do_thi_accuracy_theo_epoch.png')
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,6))
    labels=[f"H={h['hidden']} / LR={h['lr']}" for h in histories]
    score=[h['best_val_loss'] for h in histories]
    idx=np.arange(len(histories))
    ax.bar(idx,score,color='#527cb5')
    for k,v in enumerate(score):ax.text(k,v+.002,f'{v:.3f}',ha='center',fontsize=9)
    ax.set_xticks(idx,labels,rotation=35,ha='right')
    ax.set(ylabel='Minimum Validation CCE',title='Comparison of 8 ANN configurations')
    ax.grid(axis='y',alpha=.15)
    fig.tight_layout()
    fig.savefig(RESULTS/'14_so_sanh_validation_loss_8_mo_hinh.png')
    plt.close(fig)
    print('PNG: 04_so_sanh_8_do_thi_loss.png | 05_so_sanh_learning_rate.png | 06_loss_mo_hinh_tot_nhat.png')
    e=best['best_epoch']-1
    print(f"Nhan xet dinh luong: train CCE={best['train_loss'][e]:.6f}; "
          f"val CCE={best['val_loss'][e]:.6f}; "
          f"chenh lech Val-Train={best['val_loss'][e]-best['train_loss'][e]:+.6f}")
    print('5 class ngau nhien co CCE=-ln(1/5)=ln(5)=%.6f'%math.log(5))
    print('Neu loss gan ln(5) va accuracy thap => co dau hieu underfitting / dac trung chua du.')
    print('Neu train loss giam manh nhung val tang => dau hieu overfitting.')

def export_testing(records,X,y,test,best):
    section("BUOC 5 - TEST SET (75 ANH) - DO LUONG CHI TIET")
    selected=np.flatnonzero(test)
    Y=y[test]
    P=best['best_model'].forward(X[test])[3]
    pred=np.argmax(P,axis=1)
    acc=accuracy_score(Y,pred)
    labels=np.arange(5)
    cmx=confusion_matrix(Y,pred,labels=labels)
    precision,recall,f1,support=precision_recall_fscore_support(Y,pred,labels=labels,zero_division=0)
    macro_p=float(np.mean(precision)); macro_r=float(np.mean(recall));macro_f=float(np.mean(f1))
    print("Predict = argmax(softmax[0..4])")
    print("Accuracy = (so du doan dung) / N_test")
    print("Precision_i = TP_i/(TP_i+FP_i), Recall_i = TP_i/(TP_i+FN_i)")
    print("F1_i = 2*Precision_i*Recall_i/(Precision_i+Recall_i)")
    print("Macro = (tong diem 5 class)/5; denominator bang 0 -> metric=0")
    print(f"Accuracy = {int(np.sum(Y==pred))}/75 = {acc:.6f} = {acc*100:.2f}%")
    preds=[]
    for k,idx in enumerate(selected):
        r=records[int(idx)];pr=P[k]
        row={'ID':r['id'],'Anh_processed':r['processed_image'],'Nhan_that':r['class'],
             'Nhan_du_doan':int(pred[k])+1,'Dung_sai':'DUNG' if Y[k]==pred[k] else 'SAI'}
        for j in range(5):row[f'P_class_{j+1}']=f'{pr[j]:.10f}'
        preds.append(row)
        if r['class']==1 and sum(rec['Nhan_that']==1 for rec in preds)<=3:
            print(f"Vi du C1 ID={r['id']:03d}: nhan={r['class']}, du_doan={pred[k]+1}, "
                  f"Softmax={np.round(pr,4)}, {'DUNG' if Y[k]==pred[k] else 'SAI'}")
    save_csv(RESULTS/'du_doan_75_anh_test.csv',preds,list(preds[0]))
    metric_rows=[]
    for c in range(5):
        tp=int(cmx[c,c]);fn=int(cmx[c,:].sum()-tp);fp=int(cmx[:,c].sum()-tp)
        tn=int(cmx.sum()-tp-fn-fp)
        print(f"Class {c+1}: TP={tp}, FP={fp}, FN={fn}; "
              f"Precision={precision[c]:.3f}, Recall={recall[c]:.3f}, F1={f1[c]:.3f}")
        if c==0:
            print(f"Vi du Class 1: Precision={tp}/({tp}+{fp})={precision[c]:.4f}; "
                  f"Recall={tp}/({tp}+{fn})={recall[c]:.4f}; F1={f1[c]:.4f}")
        metric_rows.append({'Class':c+1,'Support':int(support[c]),'TP':tp,'FP':fp,'FN':fn,
                            'TN':tn,'Precision':f'{precision[c]:.8f}',
                            'Recall':f'{recall[c]:.8f}','F1':f'{f1[c]:.8f}'})
    metric_rows.append({'Class':'MACRO','Support':75,'TP':'','FP':'','FN':'','TN':'',
                        'Precision':f'{macro_p:.8f}','Recall':f'{macro_r:.8f}',
                        'F1':f'{macro_f:.8f}'})
    save_csv(RESULTS/'do_do_tung_class.csv',metric_rows,list(metric_rows[0]))
    print(f"Macro: Precision={macro_p:.4f}; Recall={macro_r:.4f}; F1={macro_f:.4f}")
    print("MA TRAN NHAM LAN: moi hang la nhan THAT, moi cot la nhan DU DOAN")
    print(cmx)
    fig,ax=plt.subplots(figsize=(8,7))
    ax.imshow(cmx,cmap='Blues')
    ax.set_xticks(range(5),[f'C{x}' for x in range(1,6)])
    ax.set_yticks(range(5),[f'C{x}' for x in range(1,6)])
    for i in range(5):
        for j in range(5):
            ax.text(j,i,str(cmx[i,j]),ha='center',va='center',fontsize=13,
                    color='white' if cmx[i,j]>(cmx.max()/2) else 'black')
    ax.set(xlabel='Nhan du doan',ylabel='Nhan that',title='Confusion Matrix - 75 mau Test')
    fig.colorbar(ax.images[0],ax=ax,shrink=.8);fig.tight_layout()
    fig.savefig(RESULTS/'07_confusion_matrix.png');plt.close(fig)
    cm_normalized=cmx/np.maximum(cmx.sum(axis=1,keepdims=True),1)
    fig,ax=plt.subplots(figsize=(7,6))
    im=ax.imshow(cm_normalized,cmap='Blues',vmin=0,vmax=1)
    ax.set_xticks(range(5),[f'C{i}' for i in range(1,6)])
    ax.set_yticks(range(5),[f'C{i}' for i in range(1,6)])
    for i in range(5):
        for j in range(5):
            v=cm_normalized[i,j]
            ax.text(j,i,f'{v:.1%}',ha='center',va='center',
                    color='white' if v>.55 else 'black')
    ax.set(xlabel='Predicted class',ylabel='True class',title='Hu-ANN: normalized confusion matrix')
    fig.colorbar(im,ax=ax,shrink=.75)
    fig.tight_layout()
    fig.savefig(RESULTS/'15_confusion_matrix_chuan_hoa_Hu.png')
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,6))
    xs=np.arange(5);width=.25
    for values,d,color in [(precision,-width,'#2f70b5'),(recall,0,'#da794c'),(f1,width,'#4c9c80')]:
        ax.bar(xs+d,values,width=width,label={'#2f70b5':'Precision','#da794c':'Recall',
                                             '#4c9c80':'F1'}[color],color=color)
    ax.set_xticks(xs,[f'Class {i}' for i in range(1,6)])
    ax.set_ylim(0,1.08);ax.set_ylabel('Gia tri [0,1]');ax.grid(axis='y',alpha=.2)
    ax.set_title('Precision / Recall / F1 - tung class');ax.legend();fig.tight_layout()
    fig.savefig(RESULTS/'08_so_sanh_precision_recall_f1.png');plt.close(fig)
    summary={'raw_samples':100,'processed_samples':500,'train':350,'validation':75,'test':75,
             'best_hidden':best['hidden'],'learning_rate':best['lr'],
             'best_epoch':best['best_epoch'],'validation_cce':best['best_val_loss'],
             'test_accuracy':acc,'macro_precision':macro_p,'macro_recall':macro_r,
             'macro_f1':macro_f,'confusion_matrix':cmx.tolist(),
             'per_class_precision':precision.tolist(),'per_class_recall':recall.tolist(),
             'per_class_f1':f1.tolist(),
             'train_accuracy':best['train_acc'][best['best_epoch']-1],
             'validation_accuracy':best['val_acc'][best['best_epoch']-1]}
    (RESULTS/'tong_hop_ket_qua.txt').write_text(
        "KET QUA THUC NGHIEM ANN HAND GESTURE - DATASET MOI\n"+
        "5 class * 20 anh goc = 100 anh goc\n"+
        "5 class * 100 anh mau sau xoay = 500 anh JPG (chi xoay, chua xu ly)\n"+
        "5 class * 100 anh sau xu ly = 500 anh binary\n"+
        "Train=350 / Validation=75 / Test=75\n"+
        f"Best ANN: 7-{best['hidden']}-5, eta={best['lr']}, best epoch={best['best_epoch']}\n"+
        f"Validation CCE: {best['best_val_loss']:.9f}\n"+
        f"Test Accuracy: {acc:.6f} ({int(np.sum(Y==pred))}/75)\n"+
        f"Macro Precision: {macro_p:.6f}\nMacro Recall: {macro_r:.6f}\nMacro F1: {macro_f:.6f}\n"+
        f"Confusion matrix:\n{cmx}\n"+
        "LUU Y: Khong dung Test de chon mo hinh. Ket qua thuoc dung phep chia voi seed 2026.\n"+
        "Anh goc cung nguoi/goc may co the gan trung nhau, can them du lieu doc lap neu trien khai.\n",
        encoding='utf-8')
    print("Da tao 07_confusion_matrix.png, 08_so_sanh_precision_recall_f1.png")
    print(f"\nKET QUA: Test accuracy={acc*100:.2f}%, Macro-P={macro_p:.5f}, "
          f"Macro-R={macro_r:.5f}, Macro-F1={macro_f:.5f}")
    return summary

def main():
    if not DATA.is_dir():
        raise FileNotFoundError(f"Khong tim thay thu muc {DATA}")
    RESULTS.mkdir(exist_ok=True)
    PROCESSED.mkdir(exist_ok=True)
    COLOR_AUGMENTED.mkdir(exist_ok=True)
    if '--reuse-colors' not in sys.argv:
        for folder in COLOR_AUGMENTED.glob('class_*'):
            if folder.is_dir():
                for image in folder.glob('ID_*.jpg'):
                    image.unlink()
    for folder in PROCESSED.glob("class_*"):
        if folder.is_dir():
            for image in folder.glob("ID_*.png"):
                image.unlink()
    with (RESULTS / "terminal_log.txt").open("w", encoding="utf-8") as fp:
        old = sys.stdout
        sys.stdout = TerminalAndFile(old, fp)
        try:
            fig_setup()
            print("ANN 7-H-5 | Hu Moments | Batch Gradient Descent | 200 epochs")
            print("Tien xu ly: loc mau da + GrabCut phan biet tay va bong; khong dung HOG/CNN")
            print("File Python:", Path(__file__).resolve())
            print("Thu muc dataset:", DATA.resolve())
            actual_hash, nfiles = dataset_hash(DATA)
            print("So tep goc:", nfiles)
            print("SHA256 dataset:", actual_hash)
            if actual_hash != EXPECTED_DATA_SHA256:
                print("*** CANH BAO: DU LIEU KHAC VOI DATASET.ZIP DA KIEM THU ***")
                print("Ket qua Accuracy co the khac 73.33%; kiem tra lai 100 anh goc.")
            else:
                print("KIEM TRA DU LIEU: KHOP 100 anh goc tu dataset.zip")
            (RESULTS / "kiem_tra_dataset.txt").write_text(
                "DATA=" + str(DATA.resolve()) + "\n" +
                "N_FILES=" + str(nfiles) + "\n" +
                "SHA256=" + actual_hash + "\n" +
                "KHOP_DATASET_MAU=" + str(actual_hash == EXPECTED_DATA_SHA256) + "\n",
                encoding="utf-8")
            print("Goc xoay:", ANGLES)
            rng = np.random.default_rng(SEED)
            records, preview = prep_dataset(rng)
            export_five_angles(records)
            export_color_angles(records)
            export_previews(preview)
            X, y, train, val, test = arrays_from_records(records)
            export_scatter(X, y, train)
            histories, best = train_models(X, y, train, val)
            export_losses(histories, best)
            result = export_testing(records, X, y, test, best)
            print("\nKET QUA CUOI CUNG")
            print(f"ANN 7-{best['hidden']}-5, eta={best['lr']}, checkpoint={best['best_epoch']}")
            print(f"Test accuracy: {result['test_accuracy']*100:.2f}%")
            print("Ket qua hinh anh / CSV / nhat ky: ket_qua/")
        finally:
            sys.stdout = old

if __name__ == "__main__":
    main()
