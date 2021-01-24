# nerfpp_depth
Changed version of nerfplusplus (https://github.com/Kai-46/nerfplusplus):
1. train 2 networks for a box and the scene to disentangle them and move the box around 
2. extract combined forground and background depth image
### Move objects test 
![alt text](https://github.com/sally-chen/nerfpp_depth/blob/main/move0.jpg)
![alt text](https://github.com/sally-chen/nerfpp_depth/blob/main/move1.jpg)
![alt text](https://github.com/sally-chen/nerfpp_depth/blob/main/move2.jpg)
![alt text](https://github.com/sally-chen/nerfpp_depth/blob/main/move3.jpg)


### Sample depth output
![alt text](https://github.com/sally-chen/nerfpp_depth/blob/main/depth.png)
![alt text](https://github.com/sally-chen/nerfpp_depth/blob/main/depth2.png)



How to run:
1. create a new environment with the environment.ymal file 
2. test with 
python nerfpp_infer.py --config configs/carla_box/carla_box.txt --render_splits test


# Note:
1. input notes at nerfpp_infer.py line 105 
2. output notes at nerfpp_infer.py line 76
