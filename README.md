# Visualisation-perception-2D-3D

### Several tools for visualizing 3D perception result

## Contents

- [Visualisation for kitti dataset](#visualisation-for-kitti-dataset)

```bash
pip3 install -r requirement
```

## Visualisation for kitti dataset

```bash
cd visualize_kitti
python visua.py
```

- Let uncomment `draw_lidar_simple()` in `visualize_kitti/visua.py`  
![](imgs/lidar_all.png)

- Let uncomment `show_lidar_with_boxes()` in `visualize_kitti/visua.py` 
![](imgs/lidar_with_box.png)

- Let uncomment `show_image_with_boxes()` in `visualize_kitti/visua.py` 
![](imgs/image_with_box_3d.png)
![](imgs/image_with_box_2d.png)

- Let uncomment `show_lidar_on_image()` in `visualize_kitti/visua.py` 
![](imgs/lidar_range_view.png)
![](imgs/lidar_projection_image.png)

- Let uncomment `show_lidar_topview_with_boxes()` in `visualize_kitti/visua.py` 
![](imgs/lidar_topview.png)