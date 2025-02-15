import cv2
import numpy as np
import random
import json

def parse_detections(data):
    """
    JSON文字列または辞書形式の検出結果をパースし、
    クラスごとに振り分けます。
    ※ クラスが整数の場合、mapping により "arrow", "targetb", "targetw" に変換。
    """
    if isinstance(data, str):
        data = json.loads(data)
    
    print("[PYTHON LOG] Json data:", data)
    
    result = {"arrow": [], "targetw": [], "targetb": []}
    mapping = {0: "arrow", 1: "targetb", 2: "targetw"}
    
    for pred in data["predictions"]:
        cls_val = pred["class"]
        # 整数なら mapping、文字列ならそのまま
        val = mapping.get(cls_val) if isinstance(cls_val, int) else cls_val
        if val in result:
            result[val].append({
                "x": pred["x"],
                "y": pred["y"],
                "width": pred["width"],
                "height": pred["height"]
            })
    print("[PYTHON LOG] Detections grouped by class:", result)
    
    # targetw は最も大きいもの（面積最大）を選択
    largest_targetw = None
    largest_area = 0
    for candidate in result["targetw"]:
        area = candidate["width"] * candidate["height"]
        if area > largest_area:
            largest_area = area
            largest_targetw = candidate
    result["targetw"] = largest_targetw
    print("[PYTHON LOG] Selected targetw:", result["targetw"])
    
    # targetb は、targetw に最も近いものを選択（targetw が存在する場合）
    if result["targetw"] and result["targetb"]:
        ref = result["targetw"]
        closest = None
        min_dist = float('inf')
        for candidate in result["targetb"]:
            dist = np.sqrt((ref["x"] - candidate["x"])**2 + (ref["y"] - candidate["y"])**2)
            if dist < min_dist:
                min_dist = dist
                closest = candidate
        result["targetb"] = closest
    elif result["targetb"]:
        # targetw がない場合は最も大きい targetb を選択
        largest_targetb = None
        largest_area = 0
        for candidate in result["targetb"]:
            area = candidate["width"] * candidate["height"]
            if area > largest_area:
                largest_area = area
                largest_targetb = candidate
        result["targetb"] = largest_targetb

    print("[PYTHON LOG] Final parsed detections:", result)
    return result

def calculate_corners(cx, cy, width, height):
    """
    矩形の中心(cx, cy)と幅・高さから、4隅（左上、右上、右下、左下）の座標を計算。
    """
    corners = [
        [cx - width/2, cy - height/2],  # 左上
        [cx + width/2, cy - height/2],  # 右上
        [cx + width/2, cy + height/2],  # 右下
        [cx - width/2, cy + height/2]   # 左下
    ]
    print("[PYTHON LOG] Calculated corners for (cx, cy, w, h)=({},{},{},{}): {}".format(cx, cy, width, height, corners))
    return corners

def decide_arrow_tip(Cw, Cb, corners):
    """
    白い的 (Cw) と黒い的 (Cb) の位置関係から、矢の先端候補となる角を選択する。
    ※ この例では、単純に条件分岐により a1 または a2 を返す。
    """
    a1, a2, a3, a4 = corners
    print("[PYTHON LOG] decide_arrow_tip() - Cw:", Cw, "Cb:", Cb, "Corners:", corners)
    if Cb[0] > Cw[0]:
        if Cb[1] > Cw[1]:
            return a1  # 左上
        else:
            return a1  # 左下（例として a1 を利用）
    else:
        if Cb[1] > Cw[1]:
            return a2  # 右上
        else:
            return a2  # 右下（例として a2 を利用）

def process_detections(detections, image_path, threshold=128):
    """
    detections: Kotlin 側から渡される JSON 形式（または辞書）の検出結果
    image_path: 判定に使用する画像のファイルパス
    threshold: 二値化の閾値
    
    戻り値: {"results": [判定結果のリスト]}
      -1: エラー、0: 的外、1: 的中
    """
    print("[PYTHON LOG] Starting process_detections()")
    
    # 検出結果のパース
    parsed = parse_detections(detections)
    print("[PYTHON LOG] Parsed detections:", parsed)
    
    # 再度ログ出力（必要に応じて）
    print("[PYTHON LOG] Image path:", image_path)
    
    arrow_list = parsed["arrow"]
    targetw = parsed["targetw"]
    targetb = parsed["targetb"]
    
    # 必要な情報が全て揃っていなければエラー
    if not (arrow_list and targetw and targetb):
        print("[PYTHON LOG] Missing arrow, targetw, or targetb -> returning error")
        return {"results": [-1] * len(arrow_list)}
    
    # 画像読み込み
    img = cv2.imread(image_path)
    if img is None:
        print("[PYTHON LOG] Image not loaded:", image_path)
        return {"results": [-1] * len(arrow_list)}
    print("[PYTHON LOG] img.shape:", img.shape, "img.dtype:", img.dtype)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_height, img_width = img.shape[:2]
    
    # targetb の相対座標からROIの切り出し（元のコードと同じ計算）
    x = int((targetb["x"] - targetb["width"] / 2) * img_width)
    y = int((targetb["y"] - targetb["height"] / 2) * img_height)
    w = int(targetb["width"] * img_width)
    h = int(targetb["height"] * img_height)
    print("[PYTHON LOG] ROI (x, y, w, h):", x, y, w, h)
    
    roi = gray[y:y+h, x:x+w]
    print("[PYTHON LOG] ROI shape:", roi.shape)
    if roi.size == 0:
        print("[PYTHON LOG] ROI is empty, returning error result")
        return {"results": [-1] * len(arrow_list)}
    
    # 二値化
    binary = (roi > threshold).astype(np.uint8) * 255
    print("[PYTHON LOG] Binary ROI unique values:", np.unique(binary))
    
    # エッジ検出・輪郭抽出による楕円フィッティング
    blurred = cv2.GaussianBlur(binary, (5, 5), 0)
    print("[PYTHON LOG] Blurred image shape:", blurred.shape)
    edges = cv2.Canny(blurred, 50, 150)
    points = np.column_stack(np.where(edges > 0))
    print("[PYTHON LOG] Number of edge points:", len(points))
    if len(points) < 10:
        print("[PYTHON LOG] Not enough edge points, returning error result")
        return {"results": [-1] * len(arrow_list)}
    
    ellipse = cv2.fitEllipse(points)
    print("[PYTHON LOG] Fitted ellipse:", ellipse)
    
    # targetw と targetb の情報から座標変換のためのパラメータを算出
    Xw, Yw, Ww, Hw = targetw["x"], targetw["y"], targetw["width"], targetw["height"]
    Xb, Yb, Wb, Hb = targetb["x"], targetb["y"], targetb["width"], targetb["height"]
    scale = Ww / Wb
    theta = np.radians(ellipse[2])
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    print("[PYTHON LOG] scale:", scale, "theta (deg):", ellipse[2])
    print("[PYTHON LOG] Rotation matrix:\n", rotation_matrix)
    
    results = []
    for arrow in arrow_list:
        Xa, Ya, Wa, Ha = arrow["x"], arrow["y"], arrow["width"], arrow["height"]
        corners = calculate_corners(Xa, Ya, Wa, Ha)
        arrow_tip = decide_arrow_tip([Xw, Yw], [Xb, Yb], corners)
        print("[PYTHON LOG] Processing arrow:", arrow)
        print("[PYTHON LOG] Corners:", corners)
        print("[PYTHON LOG] Arrow tip:", arrow_tip)
        
        # 黒い的(targetb)の中心を原点とした相対座標に変換
        rel = np.array([arrow_tip[0] - Xb, arrow_tip[1] - Yb])
        rotated = np.dot(rotation_matrix, rel)
        print("[PYTHON LOG] Rotated relative position:", rotated)
        
        # 元のコードでは、division by zero や不正な値を防ぐために、以下の補正を行っている
        if ellipse[1][0] * scale == 0:
            print("[PYTHON LOG] Division by zero detected, returning error result")
            return {"results": [-1] * len(arrow_list)}
        corrected = np.array([rotated[0], rotated[1] * (Ww / (ellipse[1][0] * scale))])
        final_tip = corrected + np.array([Xb, Yb])
        dist = np.linalg.norm(final_tip - np.array([Xb, Yb]))
        radius = Ww / 2
        print("[PYTHON LOG] final_tip:", final_tip, "dist:", dist, "radius:", radius)
        
        if dist <= radius:
            results.append(1)
        else:
            results.append(0)
    
    print("[PYTHON LOG] arrow_list:", arrow_list)
    print("[PYTHON LOG] targetw:", targetw)
    print("[PYTHON LOG] targetb:", targetb)
    print("[PYTHON LOG] Ellipse parameters:", ellipse)
    print("[PYTHON LOG] Final result list:", results)
    
    return {"results": results}

# ※ このコードは、元の target_judgement.py の処理内容・数式を忠実に再現しています。
#     必要に応じて、各箇所のログ出力を調整してください。
