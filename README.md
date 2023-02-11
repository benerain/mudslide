# MudSlide

This file provides a docker image for identifying the location of mudslide. The algorithm detect changes in the red band, vegetation reduction, water body and texture to output the target vector file. It is an algorithm specially provided for remote sensing images with limited band (R, G, B and N bands)

该文件提供用于识别泥石流的docker镜像，算法原理是通过检测红色波段、植被、水体和地表纹理的变化来检测泥石流的位置，输出矢量文件。是专门针对有限波段的遥感图像提供的算法(R, G, B, N波段)。

# Example
inputs：  
image_pre  
<img width="478" alt="截屏2023-02-11 下午5 55 22" src="https://user-images.githubusercontent.com/21291632/218251873-5e4283b0-d9ba-42f0-87f0-615908e0871a.png">  
image_new  
<img width="479" alt="截屏2023-02-11 下午5 55 38" src="https://user-images.githubusercontent.com/21291632/218251878-bb1b7d73-fa84-48db-9e41-877993f2dd39.png">

outputs:  
![image](https://user-images.githubusercontent.com/21291632/218251673-a36e4e66-8fb8-458f-afd1-eadaab5a5b73.png)


# Useage

## 删除镜像
```shell
docker rmi engine-studio/detect_mudslide:1.0 
```
## 打包
```shell
docker build -t engine-studio/detect_mudslide:1.0 . 
```

## 导出
```shell
docker save -o piesat.studio.detect_mudslide.tar engine-studio/detect_mudslide:1.0
```

## 加载镜像
```shell
docker load -i piesat.studio.detect_mudslide.tar
```

## 运行
```shell
Linux:
docker run --rm -i -v /Users/liyujia/Downloads/AA:/data engine-studio/detect__mudslide:1.0 -pre /data/dataset_hist.tif -new /data/dataset_new.tif -o /data/dataset_result.shp

windows:
docker run --rm -i -v /d/AA:/data engine-studio/piesat.studio.detect_mud_rock_flow:1.0 -pre /data/dataset_hist.tif -new /data/dataset_new.tif -o /data/dataset_result.shp

/Users/liyujia/Downloads/AA: 挂载路径
/data: Docker镜像中的路径
-pre: 输入灾害前的影像路径, 默认影像的波段是 红 绿 蓝 近红, 必选项
-new: 输入灾害后的影像路径, 默认影像的波段是 红 绿 蓝 近红, 必选项
-o: 输出矢量文件, 必选项
```
