# 装甲板识别程序
## 1.基本原理解释
利用降低相机曝光度，来使周围环境的光亮降低，装甲板上的光凸显出来。再利用装甲板上的光柱具有平行的特性，将其识别出来。
## 2.总体流程
2.1 将视频读取进来，拆成一帧一帧的图片，对每一帧图片进行识别

2.2 将图片由RGB转为HSV，再用opencv内置的函数inRange将灯柱的颜色抠出来（这一步排除了非此颜色的物体排除）

2.3 对上一步产生的二值图进行腐蚀膨胀等操作，消除噪点

2.4 用opencv内置函数寻找轮廓，将小面积的轮廓先排除（装甲板灯柱实际不可能这么小，如果真有这么小，说明目标距离很远，纵使识别也很难打中，因此不予考虑）

2.5 将剩余的轮廓用opencv内置函数找出最小外接矩形

2.6 判断每个轮廓两两之间是否为平行关系（核心算法）

2.7 将算法认为是平行关系的轮廓画出来，并标记相关点位，PnP测距，展示

2.8 重复上述步骤直至每一帧都识别完成
## 3.核心算法详细解释
### 3.1 boxPoints函数结果重排序算法（correctBox函数）
#### 3.1.1 重排序的原因：
opencv内置函数boxPoints能求出轮廓最小外接矩形四个点的坐标，但是它的储存会因为情况不同而顺序不同，影响后面的判断。所以要进行有格式的重排序。
#### 3.1.2 重排序详解：
先建立一个列表box，然后求四个点的中点，也就是矩形的中心，再判断四个点y坐标与中心的关系，在中心上方的两个点为up点们，在中心下方的两个点为low点们，up点们里再进行排序，x坐标小的（也就是靠左边的点）放在box的0号位，x坐标大的（也就是靠右边的点）放在box的1号位。而low点们里也要进行排序，x坐标小的（也就是靠左边的点）放在box的3号位，x坐标大的（也就是靠右边的点）放在box的2号位。
### 3.2 判断平行函数（findParallel函数）
#### 3.2.1 判断平行前的准备：
可以把装甲板看成一个平行四边形，那么识别灯柱平行就可以转化为判断是否为平行四边形。根据平行四边形的判断方法，最适合的方法是判断是否对角线相互平分。应用在这里，我是用的方法是，判断两条对角线中点的距离是否小于一定的阈值，如果小于，则说明这两个中点离得很近，也就满足此为平行四边形了。
我们两两遍历轮廓，每两个轮廓都算一下两对角线中点的距离，将这些数据以（轮廓1，轮廓2，距离）的形式存储在一个列表里，用于下一步计算。
#### 3.2.2 判断平行：
将上述列表进行重排序，以距离大小升序排列。此时会出现一种情况，A与B达成平行，B与C也是平行的，而装甲板显然不满足三个平行这种情况。因此我们选择距离最小的那个。并且我们规定一个轮廓只能出现在一组平行中。所以我们实际操作时，就进行以下操作：首先从列表中第一组平行开始，因此刚才进行了重排序，所以第一个的距离是最小的，后面依次变大，我们检查这组平行中的每一个轮廓是否已经使用过，如果两个都没使用过，就将这组平行画出来，并将这两个轮廓加入到一个已使用过的列表里，让下一组轮廓进行检查。
## 4.注意事项：
有两个视频，第一个视频是1.avi，是黄色灯光的装甲板，另一个是2.avi，是红色灯光的装甲板。因此切换视频时，不仅要改变载入视频的名字，还要改变inRange的参数，参数在代码中有注释出来。
