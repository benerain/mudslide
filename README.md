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