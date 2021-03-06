import pykitti
import open3d
import time
import argparse

if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kitti_root", default="../dataset/kitti/raw_data/", help="Checkpoint file", required=False
    )
    flags = parser.parse_args()

    basedir = flags.kitti_root
    date = "2011_09_28"
    drive = "0106"

    pcd = open3d.PointCloud()
    vis = open3d.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)

    render_option = vis.get_render_option()
    render_option.point_size = 0.1

    data = pykitti.raw(basedir, date, drive)
    to_reset_view_point = True
    while True:
        for points_with_intensity in data.velo:
            points = points_with_intensity[:, :3]
            pcd.points = open3d.Vector3dVector(points)

            vis.update_geometry()
            if to_reset_view_point:
                vis.reset_view_point(True)
                to_reset_view_point = False
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.2)

    vis.destroy_window()