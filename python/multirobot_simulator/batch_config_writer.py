import trajectory_generator as tg

import os,json

file_path, filename = os.path.split(os.path.realpath(__file__))
config_pth = os.path.join(file_path,"config","sim_config.json")

# The observers are some form of lidar, camera or monocular camera.
#   What we have is the triangulated projection 
bot_data = {
    "sensors" : {
        "lm_observer": {
            "type" : "range_bearing_bearing",   # Range, Yaw, Pitch
            "active" : False,
            "frequency" : 5,
            "range" : [[0,5],[-45,45], [-45,45]],
            "error_std" : [0.4, 2.5, 2.5]
        },
        "bot_observer": {
            "type" : "relative_pose",
            "active" : True,
            "frequency" : 5,
            "range" : [[0,100],[-180,180], [-180,180]],
            "error_std" : [0.4, 0.4, 0.4, 5]
        }, 
        "gps":  {
            "type" : "gps",
            "active" : False,
            "frequency" : 1,
            "error_std" : [2.5, 2.5, 1000]     # x, y, z
        }, 
        "bearing":  {
            "type" : "horizontal_bearing",
            "active" : False,
            "frequency" : 1,
            "error_std" : 10.0
        }
    },
    "controller": {
        "type": "quad_dron_controller_wih_yaw_control",
        "frequency": 30,
        "max_lin_vel": 3,
        "max_lin_acc": 2,
        "max_rot_vel": 90,
        "max_rot_acc": 45
    }
}

if os.path.exists(config_pth):
    with open(config_pth, "r") as f:
        config_dict = json.load(f)
        config_dict.setdefault("bots", {})
else:
    config_dict = {"bots" : {}}

for i, data in enumerate(config_dict["bots"].values()):
    for key,value in bot_data.items():
        data.update({key : value})

tg.dump_outer_list_rows(config_dict,config_pth)