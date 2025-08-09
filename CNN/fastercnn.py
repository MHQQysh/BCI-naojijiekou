import numpy as np

def generate_anchors(base_size, scales, aspect_ratios, feature_map_stride):
    """
    概念性地生成多尺度锚点。

    Args:
        base_size (int): 锚点的基础边长（例如，像素）。
        scales (list): 锚点面积的缩放因子列表。
                       例如 [128**2, 256**2, 512**2]
        aspect_ratios (list): 锚点的长宽比列表。
                              例如 [0.5, 1.0, 2.0] (对应于 1:2, 1:1, 2:1)
        feature_map_stride (int): 卷积特征图相对于原始图像的步长。
                                  例如 16 （VGG网络通常是16像素）

    Returns:
        np.array: 生成的锚点列表。每个锚点是 [x_center, y_center, width, height]
                  这里的中心和宽高都是相对于 base_size 的。
    """
    anchors = []

    # 遍历每个尺度
    for scale_area in scales:
        # 计算当前尺度下的基础边长
        base_side = np.sqrt(scale_area)

        # 遍历每个长宽比
        for ar in aspect_ratios:
            # 计算锚点的宽度和高度
            # 假设面积保持不变，或者根据论文，这里是基于原始图像上的像素面积
            # 论文中提到 "box areas of 128^2, 256^2, and 512^2 pixels"
            # 所以我们应该基于这些面积和长宽比来计算实际的w和h。
            # w * h = scale_area
            # w / h = ar  => w = ar * h
            # (ar * h) * h = scale_area => ar * h^2 = scale_area => h = sqrt(scale_area / ar)
            # w = ar * h

            h = np.round(np.sqrt(scale_area / ar))
            w = np.round(ar * h)

            # 锚点的中心通常设为 (0, 0) for relative coordinates initially
            # 最终的锚点坐标是相对于特征图上的每个点的。
            x_center = 0
            y_center = 0

            anchors.append([x_center, y_center, w, h])

    return np.array(anchors)

# 假设卷积特征图的维度（例如 38x60 for an 600x1000 image with stride 16)
feature_map_height = 38
feature_map_width = 60

# 定义锚点超参数
base_anchor_size = 1 # 理论上的基础尺寸，这里我们直接用论文提到的面积
scales = [128**2, 256**2, 512**2] # 锚点框的面积
aspect_ratios = [1/2, 1/1, 2/1] # 锚点的长宽比

# 卷积特征图相对于输入图像的步长
stride = 16 

# 1. 生成基础锚点（相对于 (0,0) 点）
base_anchors = generate_anchors(base_anchor_size, scales, aspect_ratios, stride)
print("--- 基础锚点 (x_center, y_center, width, height) ---")
print(base_anchors)
print(f"每个滑动窗口位置生成 {len(base_anchors)} 个锚点\n") # 3 scales * 3 aspect_ratios = 9

# 2. 在整个特征图上生成所有锚点
all_anchors = []
for h_idx in range(feature_map_height):
    for w_idx in range(feature_map_width):
        # 计算当前特征图位置在原始图像上的中心坐标
        # 论文中提到 "An anchor is centered at the sliding window in question"
        # 并且 "total stride for both ZF and VGG nets on the last convolutional layer is 16 pixels"
        center_x_on_image = (w_idx * stride) + (stride / 2)
        center_y_on_image = (h_idx * stride) + (stride / 2)

        for anchor in base_anchors:
            # 将基础锚点平移到当前特征图位置对应的原始图像中心
            # 锚点的 (x_center, y_center, width, height) 转换为 (x1, y1, x2, y2) 格式
            w, h = anchor[2], anchor[3]
            x1 = center_x_on_image - w / 2
            y1 = center_y_on_image - h / 2
            x2 = center_x_on_image + w / 2
            y2 = center_y_on_image + h / 2
            all_anchors.append([x1, y1, x2, y2])

print(f"--- 图像中所有生成的锚点 (仅显示前10个) ---")
# 打印前10个锚点作为示例，实际数量会非常大
for i, anchor_coords in enumerate(all_anchors[:10]):
    print(f"锚点 {i+1}: x1={int(anchor_coords[0])}, y1={int(anchor_coords[1])}, x2={int(anchor_coords[2])}, y2={int(anchor_coords[3])}")

total_anchors = feature_map_height * feature_map_width * len(base_anchors)
print(f"\n总共生成了约 {total_anchors} 个锚点 (在忽略边界锚点之前)")
print(f"对于一个典型的1000x600图像，大约有20000个锚点")

# 在实际训练中，会忽略跨越图像边界的锚点
# 并且对高度重叠的锚点进行非极大值抑制（NMS）以减少冗余
# 最终，每个图像会使用约300个提议进行检测