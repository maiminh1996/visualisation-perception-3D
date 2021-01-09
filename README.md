# Visualisation-perception-2D-3D

Several tools for visualizing 3D perception result

## Contents

- [Visualisation for kitti dataset](#visualisation-for-kitti-dataset)

## Get started

```bash
git clone https://github.com/maiminh1996/visualisation-perception-3D.git
cd visualisation-perception-3D/
pip3 install -r requirement.txt
```

## Visualisation for kitti dataset

Ref: [Explain kitti: sensors, calib, etc](https://github.com/maiminh1996/biblio-self-driving-cars/blob/master/dataset/kitti.md)

Run:

```bash
cd visualize_kitti
python visua.py
```

Let uncomment these functions in `visualize_kitti/visua.py`  

| `draw_lidar_simple()` | `show_lidar_with_boxes()` |
| :--: | :--: | 
| ![](imgs/lidar_all.png) | ![](imgs/lidar_with_box.png) |
| `show_image_with_boxes()` | `show_lidar_on_image()` | 
| ![](imgs/image_with_box_3d.png) <br/> ![](imgs/image_with_box_2d.png) | ![](imgs/lidar_range_view.png) <br/> ![](imgs/lidar_projection_image.png) |
| `show_lidar_topview_with_boxes()` | |
| ![](imgs/lidar_topview_hori.png) | |