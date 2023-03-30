import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import dearpygui.dearpygui as dpg
from scipy.spatial.transform import Rotation as R
from audio_to_face.utils.commons.hparams import set_hparams, hparams
from audio_to_face.data_util.face3d_helper import Face3DHelper

face3d_helper = Face3DHelper(use_gpu=False)


set_hparams("egs/datasets/videos/May/radnerf_torso.yaml")

from audio_to_face.tasks.radnerfs.dataset_utils import RADNeRFDataset
dataset = RADNeRFDataset("val")
idexp_lm3d_mean = dataset.idexp_lm3d_mean.reshape([68,3])
lm3d_mean = idexp_lm3d_mean / 10 + face3d_helper.key_mean_shape
lm3d_mean /= 1.5 # normalize to [-1,1]

class Landmark3D:

    def __init__(self):

        # init pose [18, 3], in [-1, 1]^3
        self.points3D = np.concatenate([lm3d_mean.numpy(), np.ones([68,1])],axis=1).reshape([68,4])

        # lines [17, 2]
        self.lines = [ 
                        # yaw
                        [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5,6], [6,7], [7,8], [8,9], [9,10], [10,11], [11,12], [12,13], [13,14], [14,15], [15,16],
                        # left brow
                        [17,18], [18,19], [19,20], [20,21], 
                        # right brow
                        [22, 23], [23,24], [24,25], [25,26],
                        # nose
                        [27,28], [28,29], [29,30], [31,32], [32,33], [33,34], [34,35],
                        # left eye
                        [36,37], [37,38], [38,39], [39,40], [40,41], [41,36],
                        # right eye
                        [42,43], [43,44], [44,45], [45,46], [46,47], [47,42],
                        # mouth
                        [48, 49], [49,50], [50,51], [51,52], [52,53], [53,54], [54,55], [55,56], [56,57], [57,58], [58,59],[59,48],
                        [48, 60], [60,61], [61,62], [62,63], [63,64], [64,65], [65,66], [66,67], [67,60], [54,64]
                      ]
        # # keypoint color [18, 3]
        # self.colors = [[0, 0, 255], [255, 0, 0], [255, 170, 0], [255, 255, 0], [255, 85, 0], [170, 255, 0], 
        #                [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], 
        #                [0, 85, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    
        self.colors = [[0,0,255] for _ in range(36)] + [[0,255,0] for _ in range(12)]+ [[255,0,0] for _ in range(20)]
        self.line_colors = [[0,0,255] for _ in range(31)] + [[0,255,0] for _ in range(12)]+ [[255,0,0] for _ in range(22)]

    def draw(self, mvp, H, W):
        # mvp: [4, 4]    

        canvas = np.zeros((H, W, 3), dtype=np.uint8)

        points2D = self.points3D @ mvp.T # [18, 4]
        points2D = points2D[:, :3] / points2D[:, 3:] # NDC in [-1, 1]

        xs = (points2D[:, 0] + 1) / 2 * H # [18]
        ys = (points2D[:, 1] + 1) / 2 * W # [18]

        # 18 points
        for i in range(len(self.points3D)):
            cv2.circle(canvas, (int(xs[i]), int(ys[i])), 4, self.colors[i], thickness=-1)

        # 17 lines
        for i in range(len(self.lines)):
            cur_canvas = canvas.copy()
            X = xs[self.lines[i]]
            Y = ys[self.lines[i]]
            mY = np.mean(Y)
            mX = np.mean(X)
            length = ((Y[0] - Y[1]) ** 2 + (X[0] - X[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))
            polygon = cv2.ellipse2Poly((int(mX), int(mY)), (int(length / 2), 4), int(angle), 0, 360, 1)
            
            cv2.fillConvexPoly(cur_canvas, polygon, self.line_colors[i])
            
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
        
        canvas = canvas.astype(np.float32) / 255
        return canvas, np.stack([xs, ys], axis=1)
        

class OrbitCamera:
    def __init__(self, W, H, r=2, fovy=60, near=0.01, far=100):
        self.W = W
        self.H = H
        self.radius = r # camera distance from center
        self.fovy = fovy # in degree
        self.near = near
        self.far = far
        self.center = np.array([0, 0, 0], dtype=np.float32) # look at this point
        self.rot = R.from_matrix(np.eye(3))
        self.up = np.array([0, 1, 0], dtype=np.float32) # need to be normalized!

    # pose
    @property
    def pose(self):
        # first move camera to radius
        res = np.eye(4, dtype=np.float32)
        res[2, 3] = self.radius # opengl convention...
        # rotate
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res
        # translate
        res[:3, 3] -= self.center
        return res

    # view
    @property
    def view(self):
        return np.linalg.inv(self.pose)
    
    # intrinsics
    @property
    def intrinsics(self):
        focal = self.H / (2 * np.tan(np.radians(self.fovy) / 2))
        return np.array([focal, focal, self.W // 2, self.H // 2], dtype=np.float32)

    # projection (perspective)
    @property
    def perspective(self):
        y = np.tan(np.radians(self.fovy) / 2)
        aspect = self.W / self.H
        return np.array([[1/(y*aspect),    0,            0,              0], 
                         [           0,  -1/y,            0,              0],
                         [           0,    0, -(self.far+self.near)/(self.far-self.near), -(2*self.far*self.near)/(self.far-self.near)], 
                         [           0,    0,           -1,              0]], dtype=np.float32)

    
    def orbit(self, dx, dy):
        # rotate along camera up/side axis!
        side = self.rot.as_matrix()[:3, 0] # why this is side --> ? # already normalized.
        rotvec_x = self.up * np.radians(-0.05 * dx)
        rotvec_y = side * np.radians(-0.05 * dy)
        self.rot = R.from_rotvec(rotvec_x) * R.from_rotvec(rotvec_y) * self.rot

    def scale(self, delta):
        self.radius *= 1.1 ** (-delta)

    def pan(self, dx, dy, dz=0):
        # pan in camera coordinate system (careful on the sensitivity!)
        self.center += 0.0005 * self.rot.as_matrix()[:3, :3] @ np.array([dx, -dy, dz])


class GUI:
    def __init__(self, opt):
        self.opt = opt
        self.W = opt.W
        self.H = opt.H
        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)

        self.skel = Landmark3D()
        
        self.render_buffer = np.zeros((self.W, self.H, 3), dtype=np.float32)
        self.need_update = True # camera moved, should reset accumulation

        self.save_path = 'pose.png'
        self.mouse_loc = np.array([0, 0])
        self.points2D = None # [18, 2]
        self.point_idx = 0
        
        dpg.create_context()
        self.register_dpg()
        self.step()
        

    def __del__(self):
        dpg.destroy_context()


    def step(self):

        if self.need_update:
        
            # mvp
            mv = self.cam.view # [4, 4]
            proj = self.cam.perspective # [4, 4]
            mvp = proj @ mv

            # render our openpose image, somehow
            self.render_buffer, self.points2D = self.skel.draw(mvp, self.H, self.W)
        
            self.need_update = False
            
            dpg.set_value("_texture", self.render_buffer)

        
    def register_dpg(self):

        ### register texture 

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(self.W, self.H, self.render_buffer, format=dpg.mvFormat_Float_rgb, tag="_texture")

        ### register window

        # the rendered image, as the primary window
        with dpg.window(label="Viewer", tag="_primary_window", width=self.W, height=self.H):
            dpg.add_image("_texture")

        dpg.set_primary_window("_primary_window", True)

        # control window
        with dpg.window(label="Control", tag="_control_window", width=-1, height=-1):

            # button theme
            with dpg.theme() as theme_button:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)
                
            def callback_save(sender, app_data):
                image = (self.render_buffer * 255).astype(np.uint8)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(self.save_path, image)
                print(f'[INFO] write image to {self.save_path}')
            
            def callback_set_save_path(sender, app_data):
                self.save_path = app_data
            
            with dpg.group(horizontal=True):
                dpg.add_button(label="save image", tag="_button_save", callback=callback_save)
                dpg.bind_item_theme("_button_save", theme_button)

                dpg.add_input_text(label="", default_value=self.save_path, callback=callback_set_save_path)

            # fov slider
            def callback_set_fovy(sender, app_data):
                self.cam.fovy = app_data
                self.need_update = True

            dpg.add_slider_int(label="FoV (vertical)", min_value=1, max_value=120, format="%d deg", default_value=self.cam.fovy, callback=callback_set_fovy)

              
        ### register camera handler

        def callback_camera_drag_rotate(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            # dx = app_data[1]
            # dy = app_data[2]

            # self.cam.orbit(dx, dy)
            self.need_update = True


        def callback_camera_wheel_scale(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            delta = app_data

            self.cam.scale(delta)
            self.need_update = True


        def callback_camera_drag_pan(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.pan(dx, dy)
            self.need_update = True

        def callback_set_mouse_loc(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            # just the pixel coordinate in image
            self.mouse_loc = np.array(app_data)

        def callback_skel_select(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return
            
            # determine the selected keypoint from mouse_loc
            if self.points2D is None: return # not prepared

            dist = np.linalg.norm(self.points2D - self.mouse_loc, axis=1) # [18]
            self.point_idx = np.argmin(dist)

        
        def callback_skel_drag(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            # 2D to 3D delta
            dx = app_data[1]
            dy = app_data[2]
        
            self.skel.points3D[self.point_idx, :3] += 0.0002 * self.cam.rot.as_matrix()[:3, :3] @ np.array([dx, -dy, 0])
            self.need_update = True


        with dpg.handler_registry():
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Left, callback=callback_camera_drag_rotate)
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Middle, callback=callback_camera_drag_pan)

            # for skeleton editing
            dpg.add_mouse_move_handler(callback=callback_set_mouse_loc)
            dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Right, callback=callback_skel_select)
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Right, callback=callback_skel_drag)

        
        dpg.create_viewport(title='pose viewer', resizable=False, width=self.W, height=self.H)
        
        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core)
        
        dpg.bind_item_theme("_primary_window", theme_no_padding)
        dpg.focus_item("_primary_window")

        dpg.setup_dearpygui()

        #dpg.show_metrics()

        dpg.show_viewport()


    def render(self):

        while dpg.is_dearpygui_running():
            self.step()
            dpg.render_dearpygui_frame()


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--W', type=int, default=512, help="GUI width")
    parser.add_argument('--H', type=int, default=512, help="GUI height")
    parser.add_argument('--radius', type=float, default=3, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=25, help="default GUI camera fovy")

    opt = parser.parse_args()

    gui = GUI(opt)
    gui.render()