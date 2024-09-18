import pandas as pd
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from data_query import PostgreSQL_connection
import WaterUsageEstimator as WUE
import WaterDemandAnalyzer as WDA
import WaterScadaAnalyzer as WSA
from adjustText import adjust_text
# import online_simulator as os  # 自己的函数，现在不用了
import build_topology as bt
import set_controls as st
import time
from datetime import timedelta
from epyt import epanet
import wntr

# 输入参数
host_input = 'localhost'
port_input = 5432
username_input = 'postgres'
password_input = '123654'
database_input = 'qingyuan_data'
start_time = '2023-03-07 00:00'  # 输入整点或整半小时时刻，注意输入格式
end_time = '2023-03-08 00:00'  # 输入整点或整半小时时刻，注意输入格式
# input_inp_file = 'ShiJiao2023_12_26.inp'
report_timestep = 1800  # 报告时间步长
hydraulic_timestep = 1800  # 水力时间步长

while True:
    # ------------------------------------------------SCADA数据处理-------------------------------------------------------
    start_time_yunxing = time.perf_counter()
    # 更新 start_time 和 end_time
    start_time = pd.to_datetime(start_time) + timedelta(hours=24)
    end_time = pd.to_datetime(end_time) + timedelta(hours=24)

    # 转换为字符串
    start_time_str = start_time.strftime('%Y-%m-%d %H:%M')
    end_time_str = end_time.strftime('%Y-%m-%d %H:%M')

    # 更新输入时间
    start_time_input = start_time_str
    end_time_input = end_time_str

    # 类实例化
    ShiJiao_scada = PostgreSQL_connection(
        host_input,
        port_input,
        username_input,
        password_input,
        database_input,
        start_time_input,
        end_time_input
    )
    # 数据库链接
    ShiJiao_scada.connect_to_PostgreSQL()
    # 查询scada表数据
    scada_u = ShiJiao_scada.query_scada_data_u(start_time_input, end_time_input)
    scada_m = ShiJiao_scada.query_scada_data_m(start_time_input, end_time_input)
    # 使用 concat 函数进行纵向合并
    df_scada = pd.concat([scada_u, scada_m], axis=0)
    # 重置索引，确保合并的 DataFrame 具有唯一的索引
    df_scada = df_scada.reset_index(drop=True)
    # scada = "C:/Users/袁伟/Desktop/qingyuan_scada_2024.1.6.xlsx"
    # df_scada.to_excel(scada, index=False)

    # scada数据重采样(部分采用远传表处理方式)
    # 第1步：正常重采样
    resample_normal_scada_1 = WSA.scada_resample_1(df_scada, start_time_input, end_time_input)
    # 采样时间序列完整化
    scada_complete_time_series = WSA.complete_time_series(resample_normal_scada_1, start_time_input, end_time_input)
    # 第2步：流量就近取值
    scada_resample_near_flow = WSA.scada_resample_2_flow(scada_complete_time_series, df_scada)
    # 第2步：压力就近取值
    scada_resample_near_pressure = WSA.scada_resample_2_pressure(scada_complete_time_series, df_scada)
    # 第3步：流量线性插值
    scada_resample_interpolate_flow = WSA.scada_resample_3_flow(scada_resample_near_flow, df_scada)
    # 第3步：压力线性插值
    scada_resample_interpolate_pressure = WSA.scada_resample_3_pressure(scada_resample_near_pressure, df_scada)
    # 第4步：把流量缺失值置为0
    scada_resample_set_NAN_flow = WSA.scada_resample_4_flow(scada_resample_interpolate_flow)
    # 第4步：把压力缺失值置为0
    scada_resample_set_NAN_pressure = WSA.scada_resample_4_pressure(scada_resample_interpolate_pressure)
    # 第5步：核查scada流量数据完整性
    scada_flow_resample = WSA.complete_time_series_verification_flow(scada_resample_set_NAN_flow)
    # 第5步：核查scada压力数据完整性
    scada_pressure_resample = WSA.complete_time_series_verification_pressure(scada_resample_set_NAN_pressure)
    # 合并数据
    scada_data = WSA.merge_flow_and_pressure(scada_resample_set_NAN_flow, scada_resample_set_NAN_pressure)
    # scada = "C:/Users/袁伟/Desktop/qingyuan_scada_2024.1.6.xlsx"
    # scada_data.to_excel(scada, index=False)
    # print("scada数据：",scada_data)

    # 根据SCADA监测数据，设置边界点（水库或泵站）水头--把监测表的监测数据赋给表关联的节点
    bian_jie_tiao_jian_pressure = WSA.get_bian_jie_tiao_jian_data(scada_data)
    # print("边界条件：",bian_jie_tiao_jian_pressure)

    # （非边界压力监测点的压力监测值和关联的节点ID---LHF）于2024.1.7yw改为：全部压力表关联节点及每时刻节点压力监测值
    scada_pressure, scada_pressure_junction_list = WSA.pressure_scada(scada_data)
    print('scada_pressure:\n', scada_pressure)
    print('scada_pressure_junction_list:\n', scada_pressure_junction_list)

    # 所有SCADA流量表每时刻监测值/所有SCADA流量表关联管道ID/入口（水库出口）SCADA流量表监测值/入口（水库出口）SCADA流量表关联管道ID
    scada_flow, scada_flow_pipe_list, scada_flow_input, scada_flow_pipe_list_input, moduan_scada_xunibiao = WSA.flow_scada(scada_data)
    print('scada_flow:\n', scada_flow)
    print('scada_flow_pipe_list:\n', scada_flow_pipe_list)
    print('scada_flow_input:\n', scada_flow_input)
    print('scada_flow_pipe_list_input:\n', scada_flow_pipe_list_input)
    print('moduan_scada_xunibiao:\n', moduan_scada_xunibiao)

    # -------------------------------------------------------营收数据处理-----------------------------------------------------

    # 类实例化
    ShiJiao_user = PostgreSQL_connection(
        host_input,
        port_input,
        username_input,
        password_input,
        database_input,
        start_time_input,
        end_time_input
    )
    # 数据库链接
    ShiJiao_user.connect_to_PostgreSQL()
    # 手抄部分-------------------------(sql查询、重采样及基本水量求解、长时间序列基本水量、存储于字典)四个部分------

    # 1.sql语句查询手抄表
    result_df_shouchao = ShiJiao_user.query_shouchao_data_demand(start_time_input)
    # file_name = "C:/Users/袁伟/Desktop/shouchao_data_full.xlsx"
    # result_df_shouchao.to_excel(file_name, index=False)
    # print("手抄表数据为：",result_df_shouchao)

    # 2.对于sql查询的手抄表进行重采样，每块表只含有一个月或两个月数据，并求基本水量，单位m3/h  -----yw
    # 手抄表重采样（只包含本次抄表周期数据）
    shouchao_resample_data=WDA.resample_shouchao_data(result_df_shouchao)
    # shouchao_base = "C:/Users/袁伟/Desktop/result_shouchao_data_2024.1.6.xlsx"
    # shouchao_resample_data.to_excel(shouchao_base, index=False)

    # 3.获取长时间序列手抄表基础水量
    # 手抄表每个时刻基本水量（每个抄表周期水量求平均）
    shouchao_long_time_series_base_data = WDA.get_long_time_series_shouchao_data(shouchao_resample_data, start_time_input, end_time_input)
    # print(shouchao_base_demand)
    # shouchao = "C:/Users/袁伟/Desktop/result_shouchao_data_fulltime_2024.1.6.xlsx"
    # shouchao_long_time_series_base_data.to_excel(shouchao, index=False)

    # 4.把长时间序手抄表基础水量按表的用水类型分组并存储于字典-----yw
    shouchao_base_demand = {}
    grouped_data = shouchao_long_time_series_base_data.groupby('cus_type_code')
    for cus_type_code, group in grouped_data:
        shouchao_base_demand[cus_type_code] = group
    # print("手抄表基本水量：", shouchao_base_demand)

    # 远传部分------------------（SQL查询、重采样、聚类手抄表用水模式）--------------------
    # sql查询远传表
    result_df_yuanchuan = ShiJiao_user.query_yuanchuan_data_demand(start_time_input, end_time_input)#一天的数据
    # yuanchuan = "C:/Users/袁伟/Desktop/result_yuanchuan_SQL_YW_2024.1.6.xlsx"
    # result_df_yuanchuan.to_excel(yuanchuan, index=False)
    # 第1步：正常重采样
    yuanchuan_resample_normal = WDA.yuanchaun_resample_1(result_df_yuanchuan, start_time_input, end_time_input)
    # yuanchuan = "C:/Users/袁伟/Desktop/result_yuanchuan_1_YW_2024.1.6.xlsx"
    # yuanchuan_resample_normal.to_excel(yuanchuan, index=False)
    # 第2步：补充缺失的重采样时间点，瞬时流量置为NAN
    yuanchuan_complete_time_series = WDA.complete_time_series(yuanchuan_resample_normal, start_time_input, end_time_input)

    # 第3步：临近点取值(目前是前后一个时间步长)
    yuanchuan_resample_near = WDA.yuanchuan_resample_2(yuanchuan_complete_time_series, result_df_yuanchuan)

    # 第4步：线性插值（一天内只要有2个数就线性插值，如果精确度不够以后以后可以设定采样窗口）
    yuanchuan_resample_interpolate = WDA.yuanchuan_resample_3(yuanchuan_resample_near, result_df_yuanchuan)

    # 第5步：流量缺失值置为0
    yuanchuan_resample_set_NAN = WDA.yuanchuan_resample_4(yuanchuan_resample_interpolate)

    # 第6步：核查时间系列完整性
    yuanchuan_demand_full = WDA.complete_time_series_verification(yuanchuan_resample_set_NAN)
    # print("远传表需水量为", yuanchuan_demand)
    # yuanchuan = "C:/Users/袁伟/Desktop/yuanchuan_demand_YW_2024.1.6.xlsx"
    # yuanchuan_demand_full.to_excel(yuanchuan, index=False)

    # 获取手抄表用水模式
    # 计算手抄表用水模式（函数中包含绘图和把模式乘子导出至桌面的代码）
    all_mode_factors = WUE.cluster_pattern(yuanchuan_demand_full, start_time_input, end_time_input)

    # 手抄表需水量-dataframe结构
    shouchao_demand_unallocated_area = WDA.shouchao_denamnd(all_mode_factors, shouchao_base_demand)
    # print("shouchao_demand_unallocated_area:", shouchao_demand_unallocated_area)
    # 待分配区域远传表
    yuanchuan_demand_unallocated_area = yuanchuan_demand_full[yuanchuan_demand_full['末端用户表为-1，未分配区域为1'] == 1]
    # print("yuanchuan_demand_unallocated_area:", yuanchuan_demand_unallocated_area)
    # yuanchuan = "C:/Users/袁伟/Desktop/yuanchuan_demand_YW_2024.1.6.xlsx"
    # yuanchuan_demand.to_excel(yuanchuan, index=False)

    # 节点需水量（每个时刻各个节点的水量）
    # 手抄表SQL查询的时候就选择了待分配区域的表
    junction_demand = WDA.junction_demand(yuanchuan_demand_unallocated_area, shouchao_demand_unallocated_area)
    # print("junction_demand为：",junction_demand)
    # 生成模拟区域内每个时刻的总用水量、每个时刻远传表的总用水量以及每个时刻的每个节点的用水量
    # df_demand_result, df_demand_yuanchuan_result, demand_grouped_clustering = WDA.process_and_analyze_data(
    # shouchao_demand_unallocated_area, yuanchuan_demand)
    # print("节点水量为：",demand_grouped_clustering)

    # 虚拟表----------------------------------yw2024.1.6
    xunibiao_data = ShiJiao_user.query_XuNiBiao()

    df = pd.DataFrame(xunibiao_data)
    # 将DataFrame转换为字典
    xunibiao_dict = df.set_index('user_id')['user_junction'].to_dict()
    # print("xunibiao_dict:",xunibiao_dict)

    # -------------------------------------------------------各片区未待分配水量-------------------------------------
    # 待分配区域手抄表之和（dataframe结构，字段：三级分区、数据采集时间、瞬时流量）
    shouchao_sum_unallocated_area = WDA.sum_area_shouchao(shouchao_demand_unallocated_area)
    # print("shouchao_sum_unallocated_area:\n",shouchao_sum_unallocated_area)

    # 待分配区域远传表之和（dataframe结构，字段：三级分区、数据采集时间、瞬时流量）
    yuanchuan_sum_unallocated_area = WDA.sum_area_yuanchuan(yuanchuan_demand_unallocated_area)
    # print("yuanchuan_sum_unallocated_area:\n",yuanchuan_sum_unallocated_area)

    # 待分配区域远传手抄水量之和
    df_shouchao = pd.DataFrame(shouchao_sum_unallocated_area)
    df_yuanchaun = pd.DataFrame(yuanchuan_sum_unallocated_area)
    # 合并DataFrame
    shouchao_and_yuanchuan_sum_unallocated_area = pd.merge(df_shouchao, df_yuanchaun, on=['三级分区', '数据采集时间'], suffixes=('_shouchao', '_yuanchuan'))

    # 计算手抄和远传瞬时流量的和
    shouchao_and_yuanchuan_sum_unallocated_area['瞬时流量'] = shouchao_and_yuanchuan_sum_unallocated_area['瞬时流量_shouchao'] + shouchao_and_yuanchuan_sum_unallocated_area['瞬时流量_yuanchuan']
    # 保留所需的列
    demand_sum_unallocated_area = shouchao_and_yuanchuan_sum_unallocated_area[['数据采集时间', '瞬时流量', '三级分区']]
    # print("demand_sum_unallocated_area:\n",demand_sum_unallocated_area)

    # 待分配区域总水
    # 查询每个片区的边界SCADA流量表
    flow_scada_fefine_unallocated_area = ShiJiao_user.query_SCADA_flow_unallocated_area()
    # 计算每个片区总水（入口-出口）dataframe结构，字段：数据采集时间、瞬时流量、三级分区
    SCADA_flow_sum_unallocated_area = WDA.sum_flow_unallocated_area(scada_data,flow_scada_fefine_unallocated_area)
    # print("SCADA_flow_sum_unallocated_area:\n", SCADA_flow_sum_unallocated_area.columns)
    # print("SCADA_flow_sum_unallocated_area:\n", SCADA_flow_sum_unallocated_area)

    # 计算每个片区待分配水（SCADA总水-远传和手抄之和），（dataframe结构，字段：三级分区、数据采集时间、瞬时流量）
    # 合并DataFrame
    df_demand_and_SCADA = pd.merge(demand_sum_unallocated_area, SCADA_flow_sum_unallocated_area, on=['三级分区', '数据采集时间'], suffixes=('_demand', '_SCADA'))

    # 计算片区总水和已经分配水的差值
    df_demand_and_SCADA['瞬时流量'] = df_demand_and_SCADA['瞬时流量_SCADA'] - df_demand_and_SCADA['瞬时流量_demand']
    # 保留所需的列
    nrw = df_demand_and_SCADA[['数据采集时间', '瞬时流量', '三级分区']]
    # 将DataFrame保存为Excel文件
    # nrw.to_excel('nrw_shijiao.xlsx', index=False)
    # print("nrw:\n",nrw)

    # 查询待分配区域节点
    unallocated_area_junction = ShiJiao_user.query_junction_unallocated_area()
    # print("unallocated_area_junction:\n", unallocated_area_junction)

    # ---------------------------------------------------从头构建管网拓扑结构------------------------------------------------
    # 类实例化
    ShiJiao_topology = PostgreSQL_connection(
        host_input,
        port_input,
        username_input,
        password_input,
        database_input,
        start_time_input,
        end_time_input
    )
    # 数据库链接
    ShiJiao_topology.connect_to_PostgreSQL()

    # 查询管网基础数据
    junction_results = ShiJiao_topology.query_junction_ShiJiao()
    pipe_results = ShiJiao_topology.query_pipe_ShiJiao()
    reservoir_results = ShiJiao_topology.query_reservoir_ShiJiao()
    tank_results = ShiJiao_topology.query_tank_ShiJiao()
    valve_results = ShiJiao_topology.query_valve_ShiJiao()
    # 对查询到的管网基础数据进行处理
    junction_data = {
        'NodeID': junction_results['id'],
        'Elevation': junction_results['elevation'],
        'X-Coord': junction_results['coordinate_x'],
        'Y-Coord': junction_results['coordinate_y']
    }
    junction_result = pd.DataFrame(junction_data)
    # 通过 Pandas DataFrame 创建 junction_info_list
    junction_info_list = list(junction_result.itertuples(index=False, name=None))

    reservoir_data = {
        'ReservoirID': reservoir_results['id'],
        'X-Coord': reservoir_results['coordinate_x'],
        'Y-Coord': reservoir_results['coordinate_y']
    }
    reservoir_result = pd.DataFrame(reservoir_data)
    # 通过 Pandas DataFrame 创建 reservoir_info_list
    reservoir_info_list = list(reservoir_result.itertuples(index=False, name=None))

    tank_data = {
        'TankID': tank_results['id'],
        'X-Coord': tank_results['coordinate_x'],
        'Y-Coord': tank_results['coordinate_y']
    }
    tank_result = pd.DataFrame(tank_data)
    # 通过 Pandas DataFrame 创建 tank_info_list
    tank_info_list = list(tank_result.itertuples(index=False, name=None))

    pipe_data = {
        'PipeID': pipe_results['id'],
        'node_1': pipe_results['from_junction'],
        'node_2': pipe_results['to_junction'],
        'diameter': pipe_results['diameter'],
        'length': pipe_results['length'],
        'roughness_coefficient': pipe_results['roughness']
    }
    pipe_result = pd.DataFrame(pipe_data)
    # 通过 Pandas DataFrame 创建 pipe_info_list
    pipe_info_list = list(pipe_result.itertuples(index=False, name=None))

    '''
    valve_PRV_data = valve_results.loc[valve_results['type'] == 'PRV']
    valve_TCV_data = valve_results.loc[valve_results['type'] == 'TCV']
    print("valve_PRV_data:\n", valve_PRV_data)
    print("valve_TCV_data:\n", valve_TCV_data)
    '''

    # 调用函数构建管网拓扑结构
    # Create an empty INP file
    testinp = 'TESTING.inp'
    d = epanet(testinp, 'CREATE')

    # Set title for project

    # Initialize epanet flow units
    d.initializeEPANET(d.setFlowUnitsGPM(), d.setOptionsHeadLossFormula('HW'))
    bt.add_junctions(d, junction_info_list)

    # 调用添加水库的函数
    bt.add_reservoirs(d, reservoir_info_list)

    # 调用添加水箱的函数
    bt.add_tanks(d, tank_info_list)

    # 调用添加管道的函数
    bt.add_pipes(d, pipe_info_list)

    # 调用函数添加阀门
    valve_data = {}
    for valve_type, group_data in valve_results.groupby('type'):
        valve_data = {
            'valveID': group_data['id'],
            'node_1': group_data['from_junction'],
            'node_2': group_data['to_junction'],
            'diameter': group_data['diameter'],
        }
        valve_result = pd.DataFrame(valve_data)
        # 通过 Pandas DataFrame 创建 pipe_info_list
        valve_info_list = list(valve_result.itertuples(index=False, name=None))
        bt.add_valves(d, valve_type, valve_info_list)

    # Save the project for future use
    d.saveInputFile(testinp)
    # d = epanet(testinp)
    # d.plot()
    # d.plot_show()
    # d.deleteProject()

    # -------------------------------------------------------在线模拟------------------------------------------------------
    start_time = datetime.datetime.strptime(start_time_input, '%Y-%m-%d %H:%M')
    end_time = datetime.datetime.strptime(end_time_input, '%Y-%m-%d %H:%M')
    current_time = start_time
    pressure_results_all_junction_average = {}  # DMA全部节点压力平均值
    pressure_results = {}  # 用于存储压力值的字典
    flow_results = {}  # 用于存储DMA内流量值的字典
    flow_results_input = {}  # 用于储存入口流量值的字典
    converted_true_scada_pressure = {
        key.to_pydatetime(): value for key, value in scada_pressure.items()
    }
    converted_true_scada_flow = {
        key.to_pydatetime(): value for key, value in scada_flow.items()
    }
    # 输入数据查看
    # print("节点需水量为：",junction_demand)
    # print(current_time, type(current_time))
    # print(junction_demand.keys(), [type(i) for i in junction_demand.keys()])
    # print("bian_jie_tiao_jian_pressure:",bian_jie_tiao_jian_pressure)
    # print("moduan_scada_xunibiao:", moduan_scada_xunibiao)
    # print("xunibiao_dict:", xunibiao_dict)
    # print("scada_flow；",scada_flow)
    absolute_diff_scada_pressure_list = []  # 添加压力误差序列
    absolute_diff_scada_flow_list = []  # 添加流量误差序列
    absolute_diff_scada_list = []  # 填加压力和流量误差序列
    time_series = []  # 添加时间序列
    sim_scada_pressure = []  # 添加模拟压力序列

    while current_time < end_time:
        # print("当前模拟时间:", current_time)
        d.deletePatternsAll()
        d.setFlowUnitsGPM()
        # 默认Demand driven analysis(DDA):d.getDemandModel().disp()
        d.setTimeHydraulicStep(hydraulic_timestep)
        # simulationDuration = (end_time - start_time).total_seconds()
        d.setTimeSimulationDuration(hydraulic_timestep)
        # 通过INP文件创建管网模型
        wn = wntr.network.WaterNetworkModel('TESTING.inp')

        # d = epanet(testinp)
        # d.plot()
        # d.plot_show()
        # {''时间点'': {'节点id': 节点需水量,...}...}
        # demand_data： {Timestamp('2023-03-08 00:00:00'): {'J02307': 0.004420424166666667,...}

        # 按待分配区域进行分组(unallocated_area_junction为dataframe，字段为：junction，area)
        unallocated_area_junction_data = unallocated_area_junction.groupby('area')
        # 把未计量水按待分配区域分组
        # nrw_data = nrw.groupby('三级分区')

        # 存储每个区域关联的节点管道长度字典的大字典
        area_pipe_lengths_dict = {}
        # 存储每个节点关联的管道长度的字典
        area_node_pipe_lengths_dict = {}
        for area, junction in unallocated_area_junction_data:
            junction_list = junction['junction'].to_list()
            # 存储管长的集合（管道ID不能重复）
            pipe_lengths_set = {}
            # 存储每个节点关联的管道长度的字典
            node_pipe_lengths_dict = {}
            # 遍历每个节点
            for node_id in junction_list:
                # 获取与节点相连的管道列表
                connected_pipes = wn.get_links_for_node(node_id)
                # 存储当前片区管道长度的字典
                current_node_pipe_lengths = {}

                # 遍历与节点相连的管道
                for pipe_id in connected_pipes:
                    # 获取管道长度
                    pipe_length = wn.get_link(pipe_id).length

                    # 将管道长度添加到集合中，以 area 作为键
                    if pipe_id not in pipe_lengths_set:
                        pipe_lengths_set[pipe_id] = pipe_length

                    # 将管道长度添加到当前节点关联的管道长度字典中
                    current_node_pipe_lengths[pipe_id] = pipe_length

                # 将当前节点关联的管道长度字典添加到总字典中
                node_pipe_lengths_dict[node_id] = current_node_pipe_lengths

            area_pipe_lengths_dict[area] = pipe_lengths_set
            area_node_pipe_lengths_dict[area] = node_pipe_lengths_dict
            # print("node_pipe_lengths_dict:\n", node_pipe_lengths_dict)
            # print("pipe_lengths_set:\n", pipe_lengths_set)
        # print("area_pipe_lengths_dict:\n", area_pipe_lengths_dict.keys())
        # print("area_node_pipe_lengths_dict:\n", area_node_pipe_lengths_dict.keys())

        # 每个片区总管长
        # 四大区域area_pipe_lengths_dict:{'DN1000待分配':
        # {'P9166': 231.238994547999, 'P02338': 68.5836, 'P7608': 112.236732418, 'P10841':
        total_pipe_length_area = {}
        for area, area_values in area_pipe_lengths_dict.items():
            total_pipe_length = sum(area_values.values())
            total_pipe_length_area[area] = total_pipe_length
        # print("total_pipe_length_area:\n", total_pipe_length_area.keys())

        # 每个片区各节点关联管道总长
        # 四大区域area_node_pipe_lengths_dict:{'DN1000待分配':
        # {'J10831-1-J9166-2': {'P9166': 231.238994547999, 'P02338': 68.5836}, 'J10841-1-J7608-2':
        # node_total_pipe_length_area：{'DN1000待分配': {'J10831-1-J9166-2': 299.822594547999, 'J10841-1-J7608-2':

        node_total_pipe_length_area = {}
        for area, area_values in area_node_pipe_lengths_dict.items():
            # node_total_pipe_length： {'J10831-1-J9166-2': 299.822594547999, 'J10841-1-J7608-2':
            node_total_pipe_length = {}
            for node, node_data in area_values.items():
                pipe_length = sum(node_data.values())
                node_total_pipe_length[node] = pipe_length

            node_total_pipe_length_area[area] = node_total_pipe_length
        # print("node_total_pipe_length_area；\n", node_total_pipe_length_area.keys())

        # 分配水量
        grouped_data = nrw.groupby('三级分区')
        # for area, area_data in area_node_pipe_lengths_dict.items():
        for area_nrw, group_data in grouped_data:
            # 设置龙塘水箱水量（由于缺少监测表，用虚拟表代替水箱，并把龙塘片区未计量水全部赋给虚拟表)
            # LongTangShuiXiang
            if area_nrw == '龙塘待分配':
                # 设置龙塘水箱水量（由于缺少监测表，用虚拟表代替水箱，并把龙塘片区未计量水全部赋给虚拟表)
                # LongTangShuiXiang
                selected_data = nrw.loc[
                    (nrw['数据采集时间'] == current_time) & (group_data['三级分区'] == area_nrw), '瞬时流量'].values
                # print("selected_data:\n" ,selected_data)

                patternID = 'LongTangShuiXiang'
                patternMult = [selected_data]
                d.addPattern(patternID, patternMult)
                longtang_shuixiang_junction = xunibiao_dict['LongTangShuiXiang']
                longtang_index = d.getNodeIndex(longtang_shuixiang_junction)
                longtang_elevation = d.getNodeElevations(longtang_index)
                d.setNodeJunctionData(longtang_index, longtang_elevation, 1, patternID)

                # wn.add_pattern('LongTangShuiXiang', selected_data)
                # shijiao_junction = wn.get_node(xunibiao_dict['LongTangShuiXiang'])
                # shijiao_junction.add_demand(base=1, pattern_name='LongTangShuiXiang')

                # 设置龙塘节点水量
                for junction, demand in junction_demand[current_time].items():
                    if junction in area_node_pipe_lengths_dict[area_nrw].keys():  # 如果节点在龙塘片区
                        patternID = junction
                        patternMult = [demand]
                        d.addPattern(patternID, patternMult)
                        junction_index = d.getNodeIndex(junction)
                        junction_elevation = d.getNodeElevations(junction_index)
                        d.setNodeJunctionData(junction_index, junction_elevation, 1, patternID)
                        '''
                        wn.add_pattern(junction, [demand])
                        shijiao_junction = wn.get_node(junction)
                        shijiao_junction.add_demand(base=1, pattern_name=junction)
                        '''
            else:  # (area == DN1000待分配、新基待分配、石岐待分配)
                current_nrw = nrw.loc[
                            (nrw['数据采集时间'] == current_time) & (nrw['三级分区'] == area_nrw), '瞬时流量'].values

                # print(f"{current_time}{area}未计量水量：\n{current_nrw}")
                # 如果未计量水量<0，则不进行未计量水量分配
                if current_nrw < 0:
                    if current_time in junction_demand.keys():
                        for junction, demand in junction_demand[current_time].items():
                            if junction in node_total_pipe_length_area[area_nrw].keys():  # 确保只分配当前区域节点水量
                                patternID = junction
                                patternMult = [demand]
                                d.addPattern(patternID, patternMult)
                                junction_index = d.getNodeIndex(junction)
                                junction_elevation = d.getNodeElevations(junction_index)
                                d.setNodeJunctionData(junction_index, junction_elevation, 1, patternID)
                                '''
                                wn.add_pattern(junction, [demand])
                                shijiao_junction = wn.get_node(junction)
                                shijiao_junction.add_demand(base=1, pattern_name=junction)
                                '''
                else:
                    if current_time in junction_demand.keys():
                        # 求该区域比流量
                        pipe_length_ratio = current_nrw / (total_pipe_length_area[area_nrw] * 3.280839)# 总管长单位有m化成ft
                        # water = sum(node_total_pipe_length_area[area].values())*0.5*pipe_length_ratio[0]
                        # print(f"{current_time}{area}未计量水量：\n{water}")

                        # 对该区域节点进行循环
                        for junction, total_pipe_length in node_total_pipe_length_area[area_nrw].items():  # 此处节点只包含当前区域
                            nrw_value = total_pipe_length * 0.5 * pipe_length_ratio[0] * 3.280839 # pipe_length_ratio是一个list，里面只有一个值,每个节点关联管长单位有m化成ft

                            # 分2种情况，第1：节点水量=水表计量水量+未计量水量，第2：节点水量=未计量水量
                            # 第1种
                            if junction in junction_demand[current_time].keys():
                                demand = junction_demand[current_time][junction] + nrw_value
                                patternID = junction
                                patternMult = [demand]
                                d.addPattern(patternID, patternMult)
                                junction_index = d.getNodeIndex(junction)
                                junction_elevation = d.getNodeElevations(junction_index)
                                d.setNodeJunctionData(junction_index, junction_elevation, 1, patternID)
                                '''
                                wn.add_pattern(junction, [demand])
                                shijiao_junction = wn.get_node(junction)
                                shijiao_junction.add_demand(base=1, pattern_name=junction)
                                '''
                            else:
                                patternID = junction
                                patternMult = [nrw_value]
                                d.addPattern(patternID, patternMult)
                                junction_index = d.getNodeIndex(junction)
                                junction_elevation = d.getNodeElevations(junction_index)
                                d.setNodeJunctionData(junction_index, junction_elevation, 1, patternID)
                                ''''
                                wn.add_pattern(junction, [nrw_value])
                                shijiao_junction = wn.get_node(junction)
                                shijiao_junction.add_demand(base=1, pattern_name=junction)
                                '''

        # 设置虚拟节点（末端只由一块SCADA表控制）
        # {’时间点‘：{’scada表表号‘：’scada表监测值‘...}...}
        # moduan_scada_flow: {Timestamp('2023-03-08 00:00:00'): {'0300005015A1': 0.0, ...}
        # {'虚拟表(scada表)的表号':’虚拟表关联节点‘}
        # xunibiao_dict: {'790000120': 'J8769-1-J8772-3',...}
        for scada_id, flow_rate in moduan_scada_xunibiao[current_time].items():
            for user_id, user_junction in xunibiao_dict.items():
                if scada_id == user_id:
                    patternID = user_id
                    patternMult = [flow_rate]
                    d.addPattern(patternID, patternMult)
                    junction_index = d.getNodeIndex(user_junction)
                    junction_elevation = d.getNodeElevations(junction_index)
                    d.setNodeJunctionData(junction_index, junction_elevation, 1, patternID)
                    '''
                    wn.add_pattern(user_id, [flow_rate])
                    shijiao_junction = wn.get_node(user_junction)
                    shijiao_junction.add_demand(base=1, pattern_name=user_id)
                    '''

        # 设置虚拟节点（末端由2块表控制）
        # {’时间点‘：{’scada表表号‘：’scada表监测值‘...}...}
        # moduan_scada_flow: {Timestamp('2023-03-08 00:00:00'): {'0300005015A1': 0.0, ...}
        # {'虚拟表(scada表)的表号':’虚拟表关联节点‘}
        # xunibiao_dict: {'790000120': 'J8769-1-J8772-3',...}

        for user_id, user_junction in xunibiao_dict.items():
            if '-' in user_id:
                start_scada, end_scada = user_id.split('-')
                # 获取对应的 flow_rate 值
                start_flow_rate = moduan_scada_xunibiao[current_time][start_scada]
                end_flow_rate = moduan_scada_xunibiao[current_time][end_scada]
                # scada表的差额
                # print("user_id",user_id)
                result_flow = abs(start_flow_rate - end_flow_rate)
                patternID = user_id
                patternMult = [result_flow]
                d.addPattern(patternID, patternMult)
                junction_index = d.getNodeIndex(user_junction)
                junction_elevation = d.getNodeElevations(junction_index)
                d.setNodeJunctionData(junction_index, junction_elevation, 1, patternID)
                '''
                wn.add_pattern(user_id, [result_flow])
                shijiao_junction = wn.get_node(user_junction)
                shijiao_junction.add_demand(base=1, pattern_name=user_id)
                '''

        if current_time in bian_jie_tiao_jian_pressure.keys():
            # reservoir_id = ['J02871', 'J14175', 'J14172', 'J02354', 'R00011']
            # reservoir_patterns = ['XinJi_Pattern', 'ShiJiaoZongGuan_Pattern', 'LongTang_Pattern',
            #                       'ShiQi_Pattern', 'ShiJiaoTiaoFeng_Pattern']
            # 水库出口压力表关联节点（与reservoir_id对应） = ['J02872', 'J146_2', 'JV4593-1', 'JV4696-2', 'YL5']
            # {'时间点': {'SCADA压力表关联节点': 压力监测值, ...}
            # head_data: {Timestamp('2023-03-08 00:00:00'): {'J02872': 47.795667, ...}
            for junction, pressure in bian_jie_tiao_jian_pressure[current_time].items():

                # 设置边界水头（水库出口压力监测点监测值+该点高程）
                if junction == 'J02872':
                    # 获取新基水库出口压力监测点关联节点高程
                    '''
                    junction = wn.get_node('J02872')
                    XinJi_elevation = junction.elevation
                    #设置水库水头
                    wn.add_pattern('XinJi_Pattern', [pressure + XinJi_elevation])
                    XinJi_reservoir = wn.get_node('J02871')
                    XinJi_reservoir.add_pattern = 'XinJi_Pattern'
                    '''
                    junction_index = d.getNodeIndex('J02872')
                    XinJi_pump_station_elevation = d.getNodeElevations(junction_index)
                    # 设置水库水头(epyt只能设置水库的elevation，不能添加模式)单位由psi换算ft
                    patternMult = (pressure / 145 * 101.477 * 3.280839) + XinJi_pump_station_elevation
                    reservoir_index = d.getNodeIndex('J02871')
                    d.setNodeElevations(reservoir_index, patternMult)

                if junction == 'J146_2':
                    # 获取石角总管水库出口压力监测点关联节点高程
                    junction_index = d.getNodeIndex('J146_2')
                    ShiJiaoZongGuan_elevation = d.getNodeElevations(junction_index)
                    # 设置水库水头
                    patternMult = (pressure / 145 * 101.477 * 3.280839) + ShiJiaoZongGuan_elevation
                    reservoir_index = d.getNodeIndex('J14175')
                    d.setNodeElevations(reservoir_index, patternMult)

                if junction == 'JV4593-1':
                    # 获取龙塘水库出口压力监测点关联节点高程
                    junction_index = d.getNodeIndex('JV4593-1')
                    LongTang_elevation = d.getNodeElevations(junction_index)
                    # 设置水库水头(单位ft)
                    patternMult = (pressure / 145 * 101.477 * 3.280839) + LongTang_elevation
                    reservoir_index = d.getNodeIndex('J14172')
                    d.setNodeElevations(reservoir_index, patternMult)

                if junction == 'JV4696-2':
                    # 获取石岐水库出口压力监测点关联节点高程
                    junction_index = d.getNodeIndex('JV4696-2')
                    ShiQi_elevation = d.getNodeElevations(junction_index)
                    # 设置水库水头
                    patternMult = (pressure / 145 * 101.477 * 3.280839) + ShiQi_elevation
                    reservoir_index = d.getNodeIndex('J02354')
                    d.setNodeElevations(reservoir_index, patternMult)

                if junction == 'YL5':
                    # 获取石角调峰水库出口压力监测点关联节点高程
                    junction_index = d.getNodeIndex('YL5')
                    ShiJiaoTiaoFeng_elevation = d.getNodeElevations(junction_index)
                    # 设置水库水头
                    patternMult = (pressure / 145 * 101.477 * 3.280839) + ShiJiaoTiaoFeng_elevation
                    reservoir_index = d.getNodeIndex('R00011')
                    d.setNodeElevations(reservoir_index, patternMult)

        # current_time_seconds = int((current_time - datetime.datetime(current_time.year, current_time.month,
                                                                     # current_time.day)).total_seconds())

        # 设置控制规则
        # 断管
        colsed_pipe_id = pipe_results.loc[pipe_results['status'] == 'Closed', 'id'].tolist()
        print("colsed_pipe_id:\n", colsed_pipe_id)
        st.set_pipe_control(d, colsed_pipe_id)

        # ######################### 阀门 (这要改，不能一个一个设置）##################################
        valve2_index = d.getLinkIndex('V00002')
        d.setLinkInitialStatus(valve2_index, 0)
        # d.setLinkInitialSetting(valve2_index, 0)
        if scada_flow[current_time]['P8982'] != 0:
            d.setLinkStatus(valve2_index, 1)

        valve4_index = d.getLinkIndex('V00004')
        d.setLinkInitialStatus(valve4_index, 0)
        # d.setLinkInitialSetting(valve4_index, 0)
        if scada_flow[current_time]['P02992'] != 0:
            d.setLinkStatus(valve4_index, 1)

        valve_PRV_index = d.getLinkIndex('V00003')
        d.setLinkInitialStatus(valve_PRV_index, 1)
        d.setLinkInitialSetting(valve_PRV_index, 50)

        valve2_index = d.getLinkIndex('V00002')
        valve2 = d.getLinkStatus(valve2_index)
        print("valve2:\n", valve2)

        valve4_index = d.getLinkIndex('V00004')
        valve4 = d.getLinkStatus(valve4_index)
        print("valve4:\n", valve4)


        p3_index = d.getLinkIndex('P10830')
        p_P10830 = d.getLinkStatus(p3_index)
        print("p_P10830:\n", p_P10830)

        p2_index = d.getLinkIndex('P431')
        p_P431 = d.getLinkStatus(p2_index)
        print("p_P431:\n", p_P431)

        p1_index = d.getLinkIndex('P9123')
        p_P9123 = d.getLinkStatus(p1_index)
        print("p_P9123:\n", p_P9123)

        d.saveInputFile(testinp)

        # Hydraulic and Quality analysis STEP-BY-STEP.
        # d.openHydraulicAnalysis()
        # d.initializeHydraulicAnalysis(0)

        # d.runHydraulicAnalysis()
        d.solveCompleteHydraulics()

        flow_index = d.getLinkIndex('P9100')
        flow = d.getLinkFlows(flow_index)
        print(f'当前模拟时间为：{current_time}\n管道P9100的模拟流量为：{flow}')

        flow1_index = d.getLinkIndex('P8772-1')
        flow = d.getLinkFlows(flow1_index)
        print(f'当前模拟时间为：{current_time}\n管道P8772-1的模拟流量为：{flow}')

        pressure_index = d.getNodeIndex('J146_2')
        pressure = d.getNodePressure(pressure_index)
        print(f'当前模拟时间为：{current_time}\n节点J146_2的模拟压力为：{pressure}')

        elevation_index = d.getNodeIndex('YL14')
        elevation_YL14 = d.getNodeElevations(elevation_index)
        print("elevation:\n", elevation_YL14)

        elevation_index = d.getNodeIndex('JV4694-1')
        elevation_JV4694 = d.getNodeElevations(elevation_index)
        print("elevation:\n", elevation_JV4694)

        diameter_index = d.getLinkIndex('P10848')
        diameter_P10848 = d.getLinkDiameter(diameter_index)
        print("diameter_P10848:\n", diameter_P10848)

        d.closeHydraulicAnalysis()

        '''
        wn.options.hydraulic.demand_model = 'DD'  # 基于流量驱动的模型
        wn.options.time.duration = (end_time - start_time).total_seconds()
        wn.options.time.report_timestep = report_timestep
        wn.options.time.hydraulic_timestep = hydraulic_timestep
        '''
        # 运行模拟
        # wn = wntr.network.WaterNetworkModel('TESTING.inp')
        # sim = wntr.sim.EpanetSimulator(wn)
        # results = sim.run_sim(version=2.2)
        '''
        node_pressure = results.node['pressure']
        pressure_at_junction_1 = node_pressure.loc[:, 'J02871']
        print('节点压力为：', pressure_at_junction_1)

        link_flowrate = results.link['flowrate']
        flow_at_pipe = link_flowrate['P10848']
        print('P10848：', flow_at_pipe.iloc[0])

        wntr.network.write_inpfile(wn, 'network_output.inp', version=2.2)
        '''
        # 获取SCADA压力表节点模拟压力和所有节点模拟压力(由psi转换成m)
        simulation_results_SCADA_pressure = {}

        for i in scada_pressure_junction_list:
            SCADA_junction_index = d.getNodeIndex(i)
            SCADA_junction_pressure = d.getNodePressure(SCADA_junction_index)
            simulation_results_SCADA_pressure[i] = SCADA_junction_pressure/145*101.477
        pressure_results[current_time] = simulation_results_SCADA_pressure

        print(f'当前模拟时间为：{current_time}\nscada压力监测节点模拟值为：{simulation_results_SCADA_pressure}')
        # print(f'当前模拟时间为：{current_time}\nscada压力监测节点监测值为：{converted_true_scada_pressure[current_time]}')

        # 获取所有SCADA流量表关联管道的模拟值(流量单位由GPM改为CMH)
        simulation_results_SCADA_flow = {}
        for i in scada_flow_pipe_list:
            SCADA_pipe_index = d.getLinkIndex(i)
            SCADA_pipe_flow = d.getLinkFlows(SCADA_pipe_index) / 4.403
            simulation_results_SCADA_flow[i] = SCADA_pipe_flow

        flow_results[current_time] = simulation_results_SCADA_flow
        print(f'当前模拟时间为：{current_time}\nscada流量监测节点模拟值为：{simulation_results_SCADA_flow}')
        print(f'当前模拟时间为：{current_time}\nscada流量监测节点监测值为：{scada_flow}')

        '''
        # 获取入口流量数据
        flow_values_input = {}
        for pipe_name, pipe_series in results.link['flowrate'].items():  # 使用 'flowrate' 替代 'flow'
            # Replace with your desired pipe names
            if pipe_name in scada_flow_pipe_list_input:
                flow_values_input[pipe_name] = pipe_series[current_time_seconds] * 3600  # 由m3/s转化为m3/h

        flow_results_input[current_time] = flow_values_input
        print(f'当前模拟时间为：{current_time}\nscada入口流量监测节点模拟值为：{flow_values_input}')
        '''

        # 计算绝对差值(压力单位：m，流量单位：m3/h)
        absolute_diff_scada_pressure = {node: (converted_true_scada_pressure[current_time][node]/145*101.477 -
                                               simulation_results_SCADA_pressure[node])
                                        for node in simulation_results_SCADA_pressure}

        print('scada_pressure绝对误差：', absolute_diff_scada_pressure)  # 单位m

        # print("converted_true_scada_flow:", converted_true_scada_flow)
        # absolute_diff_scada_flow = {pipe: abs((simulation_results_SCADA_flow[pipe] -
        # converted_true_scada_flow[current_time][pipe]) /
        # converted_true_scada_flow[current_time][pipe]) * 100
        absolute_diff_scada_flow = {pipe: abs(simulation_results_SCADA_flow[pipe] -
                                              converted_true_scada_flow[current_time][pipe]/4.403)
                                    for pipe in simulation_results_SCADA_flow}
        print('scada_flow绝对误差：', absolute_diff_scada_flow)#单位m3/h

        # 合并绝对误差字典
        absolute_diff_scada = {}
        absolute_diff_scada.update(absolute_diff_scada_pressure)
        absolute_diff_scada.update(absolute_diff_scada_flow)
        print('scada绝对误差：', absolute_diff_scada)

        '''
        # ---------------------------------------------每个时刻压力模拟结果可视化---------------------------------------
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        config = {
            "font.family": 'serif',
            "font.size": 12,
            "mathtext.fontset": 'stix',
            "font.serif": ['SimSun'],
            'axes.unicode_minus': False
        }
        plt.rcParams.update(config)  # 设置中文字体

        fig, ax = plt.subplots(figsize=(5, 3))
        #pressure = results.node['pressure']

        # 绘制网络图,节点压力为模拟时间点的模拟误差
        wntr.graphics.plot_network(wn, ax=ax, title=f'石角镇 {current_time} 压力模拟结果',

                                   node_range=[10, 55], node_colorbar_label='压力 (m)',
                                   node_size=2, node_alpha=1)

        # Add pressure values as labels on the graph
        for node_key, node_value in absolute_diff_scada_pressure.items():
            node = wn.get_node(node_key)
            # print(node, type(node))
            if node and hasattr(node, 'coordinates'):
                x, y = node.coordinates
                pressure_label = ax.text(x, y, f'{node}: {float(node_value):.2f} m', fontsize=5,
                                         color='red', verticalalignment='center')
                ax.add_artist(pressure_label)

        plt.show()

        # ------------------------------------每个时刻流量模拟结果可视化----------------------------------------------------------------
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        config = {
            "font.family": 'serif',
            "font.size": 8,
            "mathtext.fontset": 'stix',
            "font.serif": ['SimSun'],
            'axes.unicode_minus': False
        }
        plt.rcParams.update(config)  # 设置中文字体

        fig, ax = plt.subplots(figsize=(5, 3))

        # 绘制网络图,节点压力为模拟时间点的模拟误差
        wntr.graphics.plot_network(wn, ax=ax, title=f'石角镇 {current_time} 流量模拟结果',
                                    
                                    link_range=[0, 300], link_colorbar_label='流量 (m3/h)',
                                    link_width=0.5, link_alpha=1)

        # Add flowrate values as labels on the graph
        for pipe_key, pipe_value in absolute_diff_scada_flow.items():
            pipe = wn.get_link(pipe_key)
            # print(link, type(link))
            if pipe and hasattr(pipe, 'start_node') and hasattr(pipe, 'end_node'):
                start_x, start_y = wn.get_node(pipe.start_node).coordinates
                end_x, end_y = wn.get_node(pipe.end_node).coordinates
                x, y = (start_x + end_x) / 2, (start_y + end_y) / 2
                flowrate_label = ax.text(x, y, f'{pipe}: {float(pipe_value):.2f} m3/h', fontsize=4,
                                         color='red', verticalalignment='center',
                                         bbox=dict(facecolor='white', edgecolor='none', pad=0.5))
                ax.add_artist(flowrate_label)

        plt.show()
        '''
        # Append values to the lists
        sim_scada_pressure.append(simulation_results_SCADA_pressure)
        absolute_diff_scada_pressure_list.append(absolute_diff_scada_pressure)
        absolute_diff_scada_flow_list.append(absolute_diff_scada_flow)
        # absolute_diff_scada_list.append(absolute_diff_scada)

        # 前进到下一个时间步长
        current_time += timedelta(minutes=30)
        time_series.append(current_time)

    # # 生成数据管道方便画图
    # df_sim_scada_pressure = pd.DataFrame(sim_scada_pressure)
    # df_sim_scada_pressure.to_excel('df_sim_scada_pressure.xlsx', index=False)
    # df_ture_scada_pressure = pd.DataFrame(converted_true_scada_pressure).T
    # df_ture_scada_pressure = df_ture_scada_pressure/145*101.477
    # df_ture_scada_pressure.to_excel('df_ture_scada_pressure.xlsx', index=False)
    df_abs_scada_pressure = pd.DataFrame(absolute_diff_scada_pressure_list)
    # 设置 df_abs_scada_pressure 的索引为 time_series
    df_abs_scada_pressure.index = time_series
    # 将DataFrame保存为Excel文件
    df_abs_scada_pressure.to_excel('abs_scada_pressure.xlsx', index=False)
    # print('df_abs_scada_pressure:\n', df_abs_scada_pressure)
    df_abs_scada_flow = pd.DataFrame(absolute_diff_scada_flow_list)
    # print('df_abs_scada_flow:\n', df_abs_scada_flow)
    # df_abs_scada_flow.to_excel('df_abs_scada_flow_shijiao_epyt.xlsx', index=False)
    # df_abs_scada_pressure.to_excel('df_abs_scada_pressure_shijiao_epyt.xlsx', index=False)

    # ---------------------------------------压力模拟结果可视化（每天48个数取平均）------------------------------------
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    config = {
        "font.family": 'serif',
        "font.size": 12,
        "mathtext.fontset": 'stix',
        "font.serif": ['SimSun'],
        'axes.unicode_minus': False
    }
    plt.rcParams.update(config)  # 设置中文字体
    fig, ax = plt.subplots(figsize=(10, 6))
    # 计算每一列的平均值
    result_pressure_average = df_abs_scada_pressure.mean()
    # print("result_pressure_average:", result_pressure_average)
    result_pressure_average_dict = result_pressure_average.to_dict()
    # print("column_averages_dict:", result_pressure_average_dict)

    wn = wntr.network.WaterNetworkModel('TESTING.inp')
    # 绘制网络图,节点压力为模拟时间点的模拟误差
    current_date = current_time.date()
    # 往前推一天才是正确的日期
    current_date = current_date - timedelta(days=1)
    wntr.graphics.plot_network(wn, ax=ax,
                               node_attribute=result_pressure_average,
                               node_range=[10, 25], node_colorbar_label='pressure(m)',
                               node_size=20,  # 减小节点大小
                               node_alpha=1,
                               link_width=0.5)

    # 初始化一个列表来存储文本对象
    texts = []
    # Add pressure values as labels on the graph
    for node_key, node_value in result_pressure_average_dict.items():
        node = wn.get_node(node_key)
        # print(node, type(node))
        if node and hasattr(node, 'coordinates'):
            x, y = node.coordinates
            text = ax.text(x, y, f'{node.name}', fontsize=12,
                            color='green', verticalalignment='center')
            texts.append(text)

    # 使用adjust_text函数自动调整文本位置
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='black'))

    # 保存图片
    plt.savefig('WDS_plot.png', dpi=600)
    # 显示图表
    plt.show()

    # plt.pause(3)  # 暂停3秒
    # plt.close()
    '''
    # ---------------------------------------------流量模拟结果可视化（每天48个数取平均）------------------------------------------------
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    config = {
        "font.family": 'serif',
        "font.size": 12,
        "mathtext.fontset": 'stix',
        "font.serif": ['SimSun'],
        'axes.unicode_minus': False
    }
    plt.rcParams.update(config)  # 设置中文字体

    fig, ax = plt.subplots(figsize=(10, 6))
    # 计算每一列的平均值
    # print("df_abs_scada_flow:\n", df_abs_scada_flow)
    # df_abs_scada_flow.to_excel('df_abs_scada_flow.xlsx', index=False)
    result_flowrate_average = df_abs_scada_flow.mean()
    print("result_flowrate_average:\n", result_flowrate_average)
    result_flowrate_average_dict = result_flowrate_average.to_dict()
    # print("result_flowrate_average_dict:", result_flowrate_average_dict)

    # 绘制网络图,节点压力为模拟时间点的模拟误差
    current_date = current_time.date()
    current_date = current_date - timedelta(days=1)
    wntr.graphics.plot_network(wn, ax=ax, title=f'石角镇 {current_date} 流量模拟结果',
                               link_attribute=result_flowrate_average,
                               link_range=[0, 300], link_colorbar_label='流量 (%)',
                               link_width=1, link_alpha=1)

    # 初始化一个列表来存储文本对象
    texts = []
    # Add flowrate values as labels on the graph for pipes
    for pipe_key, pipe_value in result_flowrate_average_dict.items():
        pipe = wn.get_link(pipe_key)
        if pipe and hasattr(pipe, 'start_node') and hasattr(pipe, 'end_node'):
            start_x, start_y = wn.get_node(pipe.start_node).coordinates
            end_x, end_y = wn.get_node(pipe.end_node).coordinates
            x, y = (start_x + end_x) / 2, (start_y + end_y) / 2
            flowrate_label = ax.text(x, y, f'{pipe}: {float(pipe_value):.2f} %', fontsize=7,
                                     color='red', verticalalignment='center',
                                     bbox=dict(facecolor='white', edgecolor='none', pad=0.5))
            ax.add_artist(flowrate_label)
            texts.append(flowrate_label)

    # 使用adjust_text函数自动调整文本位置
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='black'))
    plt.show()
    '''
    # 将输入时间转为datetime.timedelta
    # start_time_input = datetime.datetime.strptime(start_time_input, '%Y-%m-%d %H:%M')
    # end_time_input = datetime.datetime.strptime(end_time_input, '%Y-%m-%d %H:%M')

    # 计算运行总时间
    end_time_yunxing = time.perf_counter()
    total_time = end_time_yunxing - start_time_yunxing
    print('运行总时间：', total_time)

    # 将3分钟转换为timedelta
    time_step = timedelta(minutes=3)

    # 获取秒数
    time_step_seconds = time_step.seconds

    # 计算需要暂停的秒数
    time_sleep = int(time_step_seconds - total_time)

    # 显示倒计时
    for remaining_time in range(time_sleep, 0, -1):
        print(f"Remaining time: {remaining_time} seconds", end='\r')
        time.sleep(1)

    # 显示倒计时结束
    print("Countdown complete")

    # 继续执行程序的其他部分
    print("Continue with the rest of the program...")




'''


result_nrw, leakage, unaccounted = os.count_nrw(sim_pressure_values, scada_pressure, sim_flow_values,
                                                          scada_flow, sim_flow_values_input, scada_flow_input,
                                                          sim_pressure_results_all_junction_average)

# print(leakage)
# Convert dictionaries to DataFrame
result_nrw_df = pd.DataFrame.from_dict(result_nrw, orient='index', columns=['result_nrw'])
leakage_df = pd.DataFrame.from_dict(leakage, orient='index', columns=['leakage'])
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
config = {
    "font.family": 'serif',
    "font.size": 12,
    "mathtext.fontset": 'stix',
    "font.serif": ['SimSun'],
    'axes.unicode_minus': False
}
plt.rcParams.update(config)
# Merge the two DataFrames
data = pd.merge(result_nrw_df, leakage_df, left_index=True, right_index=True)

# Set up plot
plt.figure(figsize=(10, 6))
ax = plt.gca()

# Plot result_nrw and leakage
data.plot(y=['result_nrw', 'leakage'], ax=ax)

# Set labels and title
plt.xlabel('时间戳')
plt.ylabel(r'瞬时流量 ($m^3/s$)')
# plt.title('Result NRW and Leakage')

# Format x-axis as dates
date_format = mdates.DateFormatter('%Y-%m-%d %H:%M')
ax.xaxis.set_major_formatter(date_format)

# Set x-axis tick intervals
ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))

# Rotate x-axis labels
plt.xticks(rotation=45)

# Show legend and tight layout
plt.legend(["未计量用水", "泄漏量"])
plt.tight_layout()

# Show the plot
plt.show()

#水力模拟（添加了部分未计量水量）
pressure_results_all_junction_average_add_unaccounted, pressure_results_add_unaccounted, \
    flow_results_add_unaccounted, flow_results_input_add_unaccounted, df_time_list,\
    abs_diff_scada_pressure_add_unaccounted, abs_diff_scada_flow_add_unaccounted = \
    os.run_real_time_simulation_add_unaccounted(input_inp_file, start_time_input, end_time_input, report_timestep,
                                                hydraulic_timestep,junction_demand,bian_jie_tiao_jian_pressure,
                                                unaccounted, scada_pressure, scada_flow, scada_flow_input,
                                                scada_pressure_junction_list, scada_flow_pipe_list, scada_flow_pipe_list_input)
# print('df_abs_diff_scada_pressure\n', df_abs_diff_scada_pressure)
# print('df_abs_diff_scada_flow\n', df_abs_diff_scada_flow)
# print('abs_diff_scada_pressure_add_unaccounted\n', abs_diff_scada_pressure_add_unaccounted)
# print('abs_diff_scada_flow_add_unaccounted\n', abs_diff_scada_flow_add_unaccounted)
# print('df_time_list\n', df_time_list)

# 绘制添加部分未计量用水前后scada_pressure对比图
# Get the common column names
common_columns = set(abs_diff_scada_pressure_add_unaccounted.columns) & set(df_abs_diff_scada_pressure.columns)

# Iterate over the common columns and generate plots
for column in common_columns:
    plt.figure(figsize=(10, 6))
    plt.plot(df_time_list, abs_diff_scada_pressure_add_unaccounted[column], label='添加部分未计量用水后的节点压力时间序列')
    plt.plot(df_time_list, df_abs_diff_scada_pressure[column], label='未添加部分未计量用水后的节点压力时间序列')

    # Add labels and title
    # plt.xlabel('时间戳')

    plt.ylabel('压力值（m)')
    plt.title(f'{column} SCADA压力误差对比图')

    plt.xticks(rotation=45)
    # Add legend
    plt.legend()

    # Show the plot
    plt.show()

# Get the common column names
common_columns = set(df_abs_diff_scada_flow.columns) & set(abs_diff_scada_flow_add_unaccounted.columns)

# 绘制添加部分未计量用水前后scada_flow对比图
# Iterate over the common columns and generate plots
for column in common_columns:
    plt.figure(figsize=(10, 6))
    plt.plot(df_time_list, df_abs_diff_scada_flow[column], label='未添加部分未计量用水后的管段流量时间序列')
    plt.plot(df_time_list, abs_diff_scada_flow_add_unaccounted[column], label='添加部分未计量用水后的管段流量时间序列')

    # Add labels and title
    # plt.xlabel('时间戳')
    plt.ylabel('瞬时流量 ($m^3/s$)')

    plt.title(f'{column} SCADA流量误差对比图')

    plt.xticks(rotation=45)
    # Add legend
    plt.legend()

    # Show the plot
    plt.show()
'''

