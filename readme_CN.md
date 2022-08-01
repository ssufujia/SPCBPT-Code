An unstable SPCBPT code

需求：

optix 5.0版本，更高的不支持

cuda 10.0版本以上

Cmake指定src文件夹configure，编译器为VS2015 x64，更高的VS因为不知名的原因似乎并不兼容

报错后在OPTIX_INSTALL_DIR 处填上optix安装的目录，默认形如C:/ProgramData/NVIDIA Corporation/OptiX SDK 5.0.1

CUDA_TOOLKIT_ROOT_DIR应该会自动填入CUDA的toolkit路径，默认应该是形如C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1的路径

CUDA_SDK_ROOT_DIR需要手动填上cuda samples的路径，默认形如C:/ProgramData/NVIDIA Corporation/CUDA Samples/v10.1

然后generate并open Project

如果提示升级windows SDK，拒绝

你还需要手动对一个文件hello_cuda.cu使用cuda编译

流程如下

①找到optixPathTracer项目，右键-生成依赖项-生成自定义-勾选cuda 1x.x

②将CUDA Files文件夹中的hello_cuda.cu文件右键-属性-项类型改为CUDA C/C++

随后设定optixPathtracer为启动项目，以release模式运行即可

如果提示找不到cublas.lib，请再重复一次这个流程

———————————————————————————————————————————————————————————————————————

一些算法的关键部分所对应的代码位置

### 预处理

-optixPathTracer.cpp中的pre_processing函数

* pre_processing函数本质是对train_api.data.get_data函数的一个包装，测算生成数据的时间，将得到的最优的Gamma和Q传到device（optix的GPU端的称呼）端
* 关于train_api.data.get_data
  * 预追踪用于训练的完整路径，使用的方法是LVCBPT，但是其实LVCBPT追踪得到的路径会有一定的相关性，我建议自己实现的使用用普通的path tracing可能要来的更好
  * 生成分类函数kappa（当然也就相应地生成子空间）
    * get_rough_weighted_sample函数
      * 使用classification_data_get_flat函数，从预追踪的完整路径里切出来前缀后缀子路径，权重为完整路径的贡献值/pdf
      * 使用HS_algorithm::stra_hs函数，从子路径集里采样子空间中心C，并同时给每个子路径寻找最近的聚类中心标号。这里其实使用了一个分层的寻找聚类中心的算法——但实测下来这个算法与直接根据权重采样没有优势，你可以无视这个算法的应用
    * 随后生成决策树并传到device端
      * 决策树在高度大于12时停止增长
  * 剩下的部分
    * 根据子空间给完整路径集中的每个前缀后缀子路径打上子空间ID
      * 得到Q
      * 训练得到最优的大Gamma
        * 比论文中的部分多的一个trick是，从基于贡献值的矩阵出发训练，在相同时间内能够收敛得更好一些。

### Light sub-path追踪

* host端（optix中的CPU端）
  * light_cache_process_ZGCBPT函数，其中ZGCBPT为我的算法SPCBPT的内部名称
    * light_cache_process_ZGCBPT函数在配置好默认的参数之后调用LT_trace函数追踪光子路径
    * 随后调用 MLP::data_obtain_cudaApi::LVC_process_simple来整理追踪的光源子路经，包括消除gapping，统计一些需要统计的元数据
    * 最后调用subspaces_api.buildSubspace函数来将每个光子路径放入对应的子空间中
* device端
  * path_trace_camera.cu文件中的light_vertex_launch函数为追踪光源子路径的入口程序
  * 由于GPU的特性，需要事先声明一个固定大小的内存空间来存放追踪出来的光顶点
    * 但是一条光路径能够追踪出来多少个光顶点是不确定的，算法的一般实现却要求每次迭代追踪的路径数目固定。同时作为一个高并发的任务，还要避免内存指示重合的race事件
    * 为了解决这一点，我的实现里让GPU的每个core都各自负责大内存里连续的小内存，core追踪固定数目的多个光路径，存放到相应的内存里，这样就兼顾了内存和并行问题。
    * 但是这需要对场景的平均光路径长度大小有所了解，并且申请的时候申请稍微多一点的内存（即便这会带来占位的gapping）。
      * 一般而言需要一次预追踪来估算平均路径大小，不过这理论上来说并不是什么特别耗时间才能得到的的指标，但是写起代码又挺麻烦，所以我的代码里就偷懒地把每个场景的光路径平均长度预设为SCENE_PATH_AVER的宏并为每个场景单独指定
        * 但是这个数值设小了渲染器会自动追踪更少的光路径，设大了最多也只会追踪预设数值的光路径（不过因为内存交换的原因速度会慢），所以也不是那么要紧
    * 理论上来说这里的实现非要扯皮的话会有一丁点关于unbias的争议，比如说内存大小有限的情况下，光顶点的规模超过了限制我们就直接整条路径都放弃，那么理论上来说更短的路径会更有可能被地追踪出来——但是这种争议基本上只存在在理论层面，实践上基本不需要考虑。
* 也因为实践上并行地追踪光顶点有这些条条框框，所以会需要在host端额外调用一些函数来整理并统计信息。
  * host端LVC_process_simple函数是不涉及子空间的，将所有光顶点都整理到连续的内存中的函数
    * 用thrust写的，本质上也是GPU端的操作
  * subspaces_api.buildSubspace是构建子空间的代码
    * 写于CPU上
    * 由于每个子空间在每次迭代时能够追踪到的路径数目是不同的，统一地给每个子空间分配同样的大小的空间并不合适
    * 为此我统计每个子空间在每轮迭代时追踪得到的顶点个数，然后按照子空间序号重新排列光顶点顺序（顺带一提因为同一子空间内的顺序无序，所以这是O(n)的)
    * 随后记下每个子空间的大小和起点，对N个光顶点也只需要同样规模的空间就可以建立子空间的LVC和采样分布
    * 实际上重排列的只是能够索引到每个顶点的index，被称为jumpUnit的单元就是干这个用的。原顶点还好好地保持LVC_process_simple整理后的顺序，这样就少涉及一个内存空间的声明和管理
    * 这是buildSubspace的实现
    * 后来提出了光顶点跨帧重用的改进，每个子空间的大小有了限制，重用版本的buildSubspace_LVC_reuse也就再增加了一个内存空间用于管理保存的光顶点
    * 不过为了保持接口的同一，所以jumpUnit等的使用也依旧保留

### eye sub-path追踪与连接

device端

* path_trace_camera.cu文件中的ZGCBPT_pinhole_camera函数为eye sub-path追踪和连接的入口程序
  * 在给定像素的情况下为每一个像素追踪eye sub-path
  * 初始化之后追踪，一二阶段采样最后连接，注释里大致标识了位置
  * direction_connect_ZGCBPT和connectVertex_ZGCBPT分别是对方向性光源和普通光源的连接评估代码
    * 效果为计算贡献值/（一部分的pdf）之后乘上MIS权重，pdf的另一部分是光选择部分，由负责第一二阶段采样的ZGCBPT_pinhole_camera侧代码完成

### eye sub-path 和 light sub-path 的追踪细节

打中场景后由hit program维护sub-path， 更新基本信息和MIS信息和下一次追踪的方向等

详情在hit_program.cu的BDPT_closest_hit（eye sub-path的closest hit） 和BDPT_L_closest_hit(light sub-path的closest hit)中查看

打到光源的时候会有另一套操作逻辑，在light_hit_program.cu的BDPT_closest_hit中查看



### 关于RMIS计算

在rmis.h文件中

是混沌，理解难度较高

