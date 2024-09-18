import pandas as pd
from sklearn.cluster import KMeans
import datetime
import numpy as np
#对于sql查询的手抄表进行重采样（每块表只含有一个月或两个月数据，并求基本水量，单位m3/h）-----yw
def resample_shouchao_data (df_shouchao):
    # 将时间字符串转换为 datetime 对象
    df_shouchao['上次抄表时间'] = pd.to_datetime(df_shouchao['上次抄表时间'])
    df_shouchao['本次抄表时间'] = pd.to_datetime(df_shouchao['本次抄表时间'])

    # 根据 '用户编号' 列分组，然后选择每个分组中上次抄表时间最大的那行
    shouchao_resample_data = df_shouchao.loc[df_shouchao.groupby('用户编号')['上次抄表时间'].idxmax()]

    # 把抄表水量转换成
    shouchao_resample_data['抄表水量'] = pd.to_numeric(shouchao_resample_data['抄表水量'], errors='coerce')

    # 计算每日和每小时的基础水量 GPM
    shouchao_resample_data['shouchao_base_demand_day'] = shouchao_resample_data['抄表水量'] / (
                shouchao_resample_data['本次抄表时间'] - shouchao_resample_data['上次抄表时间']).dt.days
    shouchao_resample_data['每30分钟平均用水量'] = shouchao_resample_data['shouchao_base_demand_day'] / 24 * 4.403
    return shouchao_resample_data

#对于重采样并求好基本水量的手抄表，赋予长时间序列基本数量数据------yw
def get_long_time_series_shouchao_data(df_original, start_time, end_time):
    '''
    df_resampled = pd.DataFrame()
    end_time = pd.to_datetime(end_time)
    end_time = end_time - pd.Timedelta(minutes=30)
    resampled_time_intervals = pd.date_range(start=start_time, end=end_time, freq='30T')
    for user_id, user_data in df_original.groupby('用户编号'):

        user_resampled = user_data.set_index('本次抄表时间').reindex(resampled_time_intervals, method='ffill').reset_index()
        user_resampled['数据采集时间'] = resampled_time_intervals
        user_resampled['用户编号'] = user_id
        user_resampled['user_junction'] = user_data['user_junction'].values[0]
        user_resampled['cus_type_code'] = user_data['cus_type_code'].values[0]
        user_resampled['每30分钟平均用水量'] = user_data['每30分钟平均用水量'].values[0]

        df_resampled = df_resampled.append(user_resampled)

    df_resampled = df_resampled[['数据采集时间', '每30分钟平均用水量', '用户编号', 'user_junction', 'cus_type_code']]

    return df_resampled
    '''

    df_resampled_dict = {}
    user_id_column = df_original['用户编号']
    df_resampled_dict['用户编号'] = user_id_column.to_list()

    base_demand_column = df_original['每30分钟平均用水量']
    df_resampled_dict['基本水量'] = base_demand_column.to_list()

    unallocated_area_user_column = df_original['三级分区']#-------------------------------2024.1.6
    df_resampled_dict['三级分区'] = unallocated_area_user_column.to_list()

    user_junction_column = df_original['user_junction']
    df_resampled_dict['user_junction'] = user_junction_column.to_list()

    cus_type_code_column = df_original['cus_type_code']
    df_resampled_dict['cus_type_code'] = cus_type_code_column.to_list()

    # 转换为 DataFrame
    df = pd.DataFrame(df_resampled_dict)

    # 生成时间序列
    interval = '30T'
    end_time = pd.to_datetime(end_time)
    end_time = end_time - pd.Timedelta(minutes=30)
    time_intervals = pd.date_range(start=start_time, end=end_time, freq=interval)

    # 通过笛卡尔积生成用户编号和时间序列的组合
    user_time_combinations = pd.DataFrame(
        [(user_id, user_junction, unallocated_area_user, cus_type_code, timestamp, base_water)
         for user_id, user_junction, unallocated_area_user, cus_type_code, base_water
         in zip(df['用户编号'], df['user_junction'], df['三级分区'], df['cus_type_code'], df['基本水量'])
         for timestamp in time_intervals],
        columns=['用户编号', 'user_junction', '三级分区', 'cus_type_code', '数据采集时间', '基本水量'])

    # 打印结果
    return user_time_combinations


#远传表或scada表重采样及缺失重复值处理------------------yw
#第1步，直接对sql查询数据进行重采样---------------------yw
def yuanchaun_resample_1(df_yuanchuan, start_time, end_time):
    # 将 '数据采集时间' 列转换为 datetime 对象
    df_yuanchuan['数据采集时间'] = pd.to_datetime(df_yuanchuan['数据采集时间'])
    #print(df_yuanchuan.columns)
    # 流量值全部变正
    df_yuanchuan['瞬时流量'] = pd.to_numeric(df_yuanchuan['瞬时流量'], errors='coerce')
    df_yuanchuan.loc[df_yuanchuan['瞬时流量'] < 0, '瞬时流量'] = abs(df_yuanchuan['瞬时流量'])

    # 删除重复值（数据采集时间和用户编号同时相同即认为重复）
    df_yuanchuan.drop_duplicates(subset=['数据采集时间', '用户编号'], keep='first', inplace=True)

    # 根据用户编号分组（即：逐个对每个表号的数据进行处理）
    grouped = df_yuanchuan.groupby('用户编号')
    # 初始化一个空的数据框用于存储处理后的水表数据
    df_resampled = pd.DataFrame()
    #逐块表进行重采样
    for user_id, user_data in grouped:
        # 将 '数据采集时间' 列转换为日期时间格式
        user_data['数据采集时间'] = pd.to_datetime(user_data['数据采集时间'])
        # 第1步：正常取值
        # 正常值的采样过程，设置重采样时间范围（当日00:00-23:30）
        filtered_data = user_data[(user_data['数据采集时间'] >= start_time) & (user_data['数据采集时间'] < end_time)]
        #print(filtered_data)
        # 将筛选后的数据设置为索引,并进行重采样（时间步长为30min，缺失值填充NAN）
        filtered_data = filtered_data.set_index('数据采集时间')

        resampled_data = filtered_data.resample('30T').asfreq()
        # 将未采样到的瞬时流量置为 NaN
        resampled_data.loc[resampled_data['瞬时流量'].isnull(), '瞬时流量'] = np.nan
        resampled_data = resampled_data.reset_index()
        # 提取用户编号
        resampled_data['用户编号'] = user_id
        # 提取关联节点
        user_junction = user_data['user_junction'].iloc[0]
        resampled_data['user_junction'] = user_junction
        user_type = user_data['cus_type_code'].iloc[0]
        resampled_data['cus_type_code'] = user_type
        user_area = user_data['三级分区'].iloc[0]#--------------------------------2024.1.6
        resampled_data['三级分区'] = user_area
        user_allocated = user_data['末端用户表为-1，未分配区域为1'].iloc[0]
        resampled_data['末端用户表为-1，未分配区域为1'] = user_allocated


        # 把索引列改为普通列
        resampled_data.reset_index(inplace=True)
        # 把所有数据存放在同一个数据框
        df_resampled = df_resampled.append(resampled_data)
        # 最后结果保留的列
        df_resampled = df_resampled[['数据采集时间', '瞬时流量', '用户编号', 'user_junction', 'cus_type_code', '三级分区', '末端用户表为-1，未分配区域为1']]#-----------2024.1.6
    return df_resampled
#重采样不会补全缺失的采样点，本函数作用在于补全时间序列-------------------yw
def complete_time_series(df_resampled, start_time, end_time):
    grouped = df_resampled.groupby('用户编号')
    yuanchuan_complete_time_series = df_resampled.copy()
    end_time = pd.to_datetime(end_time) - pd.Timedelta(minutes=30)  # 从结束时间减去半小时
    resampled_time_intervals = pd.date_range(start=start_time, end=end_time, freq='30T')
    #print("采样时间序列为：", resampled_time_intervals)
    for user_id, user_data in grouped:
        if len(user_data['数据采集时间']) != 48:
            missing_time_intervals = resampled_time_intervals.difference(user_data['数据采集时间'])
            #print('缺失的时间序列',missing_time_intervals)
            # 创建新DataFrame，包含缺失的数据
            df_resampled_complete = pd.DataFrame({
                '数据采集时间': missing_time_intervals,
                '瞬时流量': np.nan,
                '用户编号': user_id,
                'user_junction': user_data['user_junction'].values[0],
                'cus_type_code': user_data['cus_type_code'].values[0],
                '三级分区': user_data['三级分区'].values[0],#-----------------------------------------------------2024.1.6
                '末端用户表为-1，未分配区域为1': user_data['末端用户表为-1，未分配区域为1'].values[0],
            })
            merged_df = pd.concat([df_resampled_complete, yuanchuan_complete_time_series], ignore_index=True)
            yuanchuan_complete_time_series = merged_df
            #print("缺失的值汇总为：", df_resampled_complete)
    return yuanchuan_complete_time_series

#第2步，在邻近步长进行取值-------------以790000260和780000333为例核查无误-----yw
def yuanchuan_resample_2(resampled_data, df_yuanchuan):
    # 将 '数据采集时间' 列转换为 datetime 对象
    df_yuanchuan['数据采集时间'] = pd.to_datetime(df_yuanchuan['数据采集时间'])

    # 流量值全部变正
    df_yuanchuan['瞬时流量'] = pd.to_numeric(df_yuanchuan['瞬时流量'], errors='coerce')
    df_yuanchuan.loc[df_yuanchuan['瞬时流量'] < 0, '瞬时流量'] = abs(df_yuanchuan['瞬时流量'])

    # 删除重复值（数据采集时间和用户编号同时相同即认为重复）
    df_yuanchuan.drop_duplicates(subset=['数据采集时间', '用户编号'], keep='first', inplace=True)
    # 获取瞬时流量为空的所有行

    data_missing = resampled_data[resampled_data['瞬时流量'].isnull()]

    #print("远传数据为：", df_yuanchuan)
    #output_file = "C:/Users/袁伟/Desktop/ddf_yuanchuan_2222.xlsx"#----------------正确
    #data_missing.to_excel(output_file, index=False)

    yuanchuan_resample_near = resampled_data.copy()
    #判断是否有缺失值
    if data_missing.empty:
        print("第1步正常采样后，所有远传数据完整")
    else:
        print("第1步正常重采样后还有数据缺失，需要进行第2步进行就近取值")

        #把缺失数据按编号进行分组
        grouped = data_missing.groupby('用户编号')
        time_window = pd.Timedelta('30T')
        for user_id, user_data in grouped:
           # 获取原始瞬时流量数据不为空的行
            yuanchuan_data = df_yuanchuan[(df_yuanchuan['用户编号'] == user_id) &
                                               (~df_yuanchuan['瞬时流量'].isnull())][['用户编号', '数据采集时间', '瞬时流量']]
            #print("远传数据为:",yuanchuan_data)
            #print(user_id)
            #output_file = "C:/Users/袁伟/Desktop/yuanchuan_data_1111.xlsx"#----------------正确
            #yuanchuan_data.to_excel(output_file, index=False)
            #print(yuanchuan_data)

           # 转换为时间格式
            data_missing_times = user_data['数据采集时间']
            #print(user_id)
            #print(data_missing_times)
            for data_missing_time in data_missing_times:
                # 筛选出特定用户的数据
                start_time = data_missing_time - time_window
                end_time = data_missing_time + time_window
                user_data_near = yuanchuan_data[(yuanchuan_data['用户编号'] == user_id) &
                                        (yuanchuan_data['数据采集时间'] > start_time) &
                                        (yuanchuan_data['数据采集时间'] < end_time)]
                #print("最接近的内容：", user_data_near)
                if not user_data_near.empty:
                    # 计算时间最接近的数据点（要注意可能有多个最接近时间点，目前这里取第一个）
                    closest_time_index = user_data_near['数据采集时间'].sub(data_missing_time).abs().idxmin()
                    closest_time = user_data_near.loc[closest_time_index, '数据采集时间']

                    closest_value = user_data_near.loc[closest_time_index, '瞬时流量']

                    yuanchuan_resample_near.loc[(yuanchuan_resample_near['数据采集时间'] == data_missing_time) &
                                                (yuanchuan_resample_near['用户编号'] == user_id), '瞬时流量'] = closest_value
                else:
                    yuanchuan_resample_near.loc[(yuanchuan_resample_near['数据采集时间'] == data_missing_time) &
                                                (yuanchuan_resample_near['用户编号'] == user_id), '瞬时流量'] = np.nan
    return yuanchuan_resample_near

#线性插值函数(yuanchuan_resample_3中调用了)------------------------------yw
def linear_interpolation(time_1, time_2, value_1, value_2,target_time):
    # 计算时间差值
    time_difference = (target_time - time_1).total_seconds()
    total_time_range = (time_2 - time_1).total_seconds()

    # 进行线性插值
    interpolated_value = value_1 + (time_difference * (value_2 - value_1) / total_time_range)
    return interpolated_value

#第3步，线性插值------------------------------------yw
def yuanchuan_resample_3(resampled_data, df_yuanchuan):
    df_yuanchuan['数据采集时间'] = pd.to_datetime(df_yuanchuan['数据采集时间'])
    df_yuanchuan['瞬时流量'] = pd.to_numeric(df_yuanchuan['瞬时流量'], errors='coerce')
    df_yuanchuan.loc[df_yuanchuan['瞬时流量'] < 0, '瞬时流量'] = abs(df_yuanchuan['瞬时流量'])
    df_yuanchuan.drop_duplicates(subset=['数据采集时间', '用户编号'], keep='first', inplace=True)
    #获取进行第1、2步之后瞬时流量为空的数据
    data_missing = resampled_data[resampled_data['瞬时流量'].isnull()]

    if data_missing.empty:
        print("第2步就近取值后，远传数据已经完整")
    else:
        print("第2步就近取值后，还有数据缺失，需要进行第3步线性插值")
        grouped = data_missing.groupby('用户编号')

        for user_id, user_data in grouped:
            #获取原始瞬时流量数据不为空的行
            yuanchuan_data_no_NAN = df_yuanchuan[(df_yuanchuan['用户编号'] == user_id) &
                                                   (~df_yuanchuan['瞬时流量'].isnull())][['用户编号', '数据采集时间', '瞬时流量']]

            data_missing_times = user_data['数据采集时间']

            for data_missing_time in data_missing_times:
                closest_times = yuanchuan_data_no_NAN.loc[yuanchuan_data_no_NAN['用户编号'] == user_id, '数据采集时间']\
                    .sort_values(key=lambda x: abs(x - data_missing_time)).head(2)
                #判断邻近时间点是否有2个（保证时间点和流量值都存在）
                #print(user_id)
                if len(closest_times) == 2:
                    time_1 = closest_times.iloc[0]  # 获取第一个时间点
                    time_2 = closest_times.iloc[1]  # 获取第二个时间点
                    #print("插值时间1为：",time_1)
                    #print("插值时间2为：",time_2)
                    # 把数据采集时间设置为索引
                    yuanchuan_data_no_NAN.set_index('数据采集时间', inplace=True)
                    value_1 = yuanchuan_data_no_NAN.loc[time_1, '瞬时流量']
                    value_2 = yuanchuan_data_no_NAN.loc[time_2, '瞬时流量']

                    #print("数值1为：",value_1)
                    #print("数值2为：",value_2)

                    # 调用线性插值函数并输出结果
                    desire_value = linear_interpolation(time_1, time_2, value_1, value_2, data_missing_time)
                    resampled_data.loc[(resampled_data['数据采集时间'] == data_missing_time) &
                                        (resampled_data['用户编号'] == user_id), '瞬时流量'] = desire_value
                    #把数据采集时间回复普通列（不然前面closest_times处会出问题）
                    yuanchuan_data_no_NAN.reset_index(inplace=True)
                else:
                    print("时间点少于2个，无法插值")
                    resampled_data.loc[(resampled_data['数据采集时间'] == data_missing_time) &
                                       (resampled_data['用户编号'] == user_id), '瞬时流量'] = np.nan
    return resampled_data
#第4步：瞬时流量为空的置0-------------------------------------------yw
def yuanchuan_resample_4(resampled_data):
    #因为线性插值条件较松，插值后可能出现负值
    resampled_data['数据采集时间'] = pd.to_datetime(resampled_data['数据采集时间'])
    resampled_data['瞬时流量'] = pd.to_numeric(resampled_data['瞬时流量'], errors='coerce')
    resampled_data.loc[resampled_data['瞬时流量'] < 0, '瞬时流量'] = abs(resampled_data['瞬时流量'])
    resampled_data.drop_duplicates(subset=['数据采集时间', '用户编号'], keep='first', inplace=True)

    data_missing = resampled_data[resampled_data['瞬时流量'].isnull()]
    # 判断是否有缺失值
    if data_missing.empty:
        print("第3步线性插值后，所有远传数据完整")
    else:
        print("第3步线性插值后，还有数据缺失（有数据不能插值）")
        grouped = data_missing.groupby('用户编号')
        for user_id, user_data in grouped:
            data_missing_times = user_data['数据采集时间']
            for data_missing_time in data_missing_times:
                resampled_data.loc[(resampled_data['数据采集时间'] == data_missing_time) &
                                   (resampled_data['用户编号'] == user_id), '瞬时流量'] = 0
    return resampled_data

#判断所有时间序列数据是否齐全（逐块表进行核查）---------------------------------------yw
def complete_time_series_verification(resampled_data):
    grouped = resampled_data.groupby('用户编号')
    for user_id, user_data in grouped:
        # 判断每块表数据是否为48
        if len(user_data['瞬时流量']) != 48:
            print("缺失的用户编号：", user_id)
            raise ValueError("远传数据缺失")

    #把流量单位换成m3/s
    #resampled_data['瞬时流量'] = resampled_data['瞬时流量']/3600
    #把流量单位换成GPM
    resampled_data['瞬时流量'] = resampled_data['瞬时流量'] * 4.403
    #print("经过数据处理后，所有远传数据均完整")
    return resampled_data



#手抄表基本水量*模式乘子=手抄表需水量---------------------------------yw
def shouchao_denamnd(all_mode_factors, shouchao_base_demand):
    merged_dict = {}

    # 遍历不同的类别键
    for key in all_mode_factors:
        # 检查该类别是否存在于基础用水量字典中
        if key in shouchao_base_demand:
            # 根据数据采集时间将模式因子和基础用水量进行合并
            merged_df = pd.merge(all_mode_factors[key], shouchao_base_demand[key], on='数据采集时间')

            # 计算瞬时流量：模式因子 * 每30分钟平均用水量
            merged_df['瞬时流量'] = merged_df['ModeFactor'] * merged_df['基本水量']
            merged_dict[key] = merged_df

    # 将 merged_dict 中的所有值合并为一个大的 DataFrame
    merged_values = list(merged_dict.values())
    merged_dataframe = pd.concat(merged_values, axis=0, ignore_index=True)
    # 从 merged_dataframe 中选择需要的列
    selected_columns = ['数据采集时间', '瞬时流量', '用户编号', 'user_junction', '三级分区']#-----------------------2024.1.6
    selected_merged_dataframe = merged_dataframe[selected_columns]
    return selected_merged_dataframe
    # print("节点水量为：", selected_merged_dataframe)

#聚合节点水量---------------------------------yw
def junction_demand(selected_merged_dataframe, yuanchuan_demand):
    # 把手抄和远传合并到一个dataframe中
    junction_demand = pd.concat([selected_merged_dataframe, yuanchuan_demand], axis=0, ignore_index=True)
    # print("总水量为：", demand)

    # 使用 groupby 对合并后的结果进行聚类，获取不同数据采集时间下的 user_junction 和 瞬时流量的总和
    grouped_demand_with_junction = junction_demand.groupby(['数据采集时间', 'user_junction'])[
        '瞬时流量'].sum().reset_index()
    # 根据 user_junction 进行聚类
    grouped_demand_clusters = grouped_demand_with_junction.groupby('user_junction')

    # print("分组聚类结果：", grouped_demand_clusters)
    # 创建一个字典来存储聚类结果
    clustering_results = {}
    # 遍历每个聚类并获取时间序列数据
    for junction, group in grouped_demand_clusters:
        clustering_results[junction] = group[['数据采集时间', '瞬时流量']].copy()
    # print("每个节点各个时刻流量为：", clustering_results)

    grouped_clustering_results = {}
    for junction, data in clustering_results.items():
        for index, row in data.iterrows():
            timestamp = row['数据采集时间']
            if timestamp not in grouped_clustering_results:
                grouped_clustering_results[timestamp] = {}
            if junction not in grouped_clustering_results[timestamp]:
                grouped_clustering_results[timestamp][junction] = 0
            grouped_clustering_results[timestamp][junction] += row['瞬时流量']

    # print("每个时刻各个节点流量为：", grouped_clustering_results)
    return grouped_clustering_results

#----------------------------------------计算各待分配区域未计量水量（总水-末端-手抄-远传）--------------------------------------------

#各待分配区域手抄表之和
def sum_area_shouchao(shouchao_data_unallocated_area):
    # 转换"数据采集时间"列的数据类型为datetime
    shouchao_data_unallocated_area['数据采集时间'] = pd.to_datetime(shouchao_data_unallocated_area['数据采集时间'])
    # 按照"三级分区"分组，并对"瞬时流量"进行求和
    shouchao_sum_unallocated_area = shouchao_data_unallocated_area.groupby(['三级分区', '数据采集时间'])['瞬时流量'].sum().reset_index()
    return shouchao_sum_unallocated_area

#各待分配区域远传表之和
def sum_area_yuanchuan(yuanchuan_data_unallocated_area):
    # 转换"数据采集时间"列的数据类型为datetime
    yuanchuan_data_unallocated_area['数据采集时间'] = pd.to_datetime(yuanchuan_data_unallocated_area['数据采集时间'])
    # 按照"三级分区"分组，并对"瞬时流量"进行求和
    yuanchuan_sum_unallocated_area = yuanchuan_data_unallocated_area.groupby(['三级分区', '数据采集时间'])[
        '瞬时流量'].sum().reset_index()
    return yuanchuan_sum_unallocated_area

#各待分配区域总水
def sum_flow_unallocated_area(scada_data,flow_scada_fefine):
    flow_scada_fefine['access_code(进水为1，出水为-1)'] = pd.to_numeric(flow_scada_fefine['access_code(进水为1，出水为-1)'])

    # 创建一个字典用于存储每个未分配区域的总水量
    results_sum_flow_unallocated = pd.DataFrame()

    flow_unallocated_area = flow_scada_fefine.groupby('待分配区域')

    # 必须要先分组，再合并，因为不同片区边界会有表重复
    for area, area_data in flow_unallocated_area:
        # 使用 merge 函数将两个数据框合并，根据 scada_id 列进行匹配
        merged_data = pd.merge(scada_data, area_data, left_on='scada_id', right_on='SCADA_id')

        # 将 '数据采集时间' 列转换为 datetime 类型
        merged_data['数据采集时间'] = pd.to_datetime(merged_data['数据采集时间'])

        # 选择 access_code 为 1 的数据
        inflow_data = merged_data[merged_data['access_code(进水为1，出水为-1)'] == 1]

        # 选择 access_code 为 -1 的数据
        outflow_data = merged_data[merged_data['access_code(进水为1，出水为-1)'] == -1]

        # 计算 access_code 为 1 和 -1 的数据采集时间的瞬时流量之和
        inflow_sum = inflow_data.groupby('数据采集时间')['瞬时流量'].sum()
        outflow_sum = outflow_data.groupby('数据采集时间')['瞬时流量'].sum()

        # 计算瞬时流量之差
        result = inflow_sum - outflow_sum
        result_df = result.reset_index().rename(columns={'瞬时流量': area})

        # 将每个未分配区域的结果添加到总结果中
        if results_sum_flow_unallocated.empty:
            results_sum_flow_unallocated = result_df
        else:
            results_sum_flow_unallocated = pd.merge(results_sum_flow_unallocated, result_df, on='数据采集时间', how='outer')

    # 将结果按照 '数据采集时间' 列排序
    results_sum_flow_unallocated = results_sum_flow_unallocated.sort_values(by='数据采集时间')
    df = pd.DataFrame(results_sum_flow_unallocated)
    # 数据重塑
    df_melted = pd.melt(df, id_vars=['数据采集时间'], var_name='三级分区', value_name='瞬时流量')
    #print("results_sum_flow_unallocated:\n", results_sum_flow_unallocated)
    return df_melted
















#LHF
'''
def run_analyzer(self, start_time, end_time):
    # 主函数，执行整个类流程

    # 将输入时间转为datetime.timedelta
    start_time = datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M')
    end_time = datetime.datetime.strptime(end_time, '%Y-%m-%d %H:%M')

    shouchao_base_demand = self.query_and_resample_data(start_time, end_time)

    yuanchuan_demand = self.query_and_resample_yuanchuan_data(start_time, end_time).sort_values(by='数据采集时间')

    # # 远传表数据查看
    # print("远传表重采样后的数据:")
    # print(yuanchuan_demand.to_string(), yuanchuan_demand.shape)
    # # 检查是否存在空值
    # print("每一个时刻各个远传表用水量是否存在空值：")
    # print(yuanchuan_demand['瞬时流量'].isnull().any())
    # # 使用 groupby 对数据采集时间进行分组，并获取每个时间点对应的行数
    # rows_per_time = yuanchuan_demand.groupby('数据采集时间').size()
    # print(rows_per_time)
    # # Print the maximum and minimum row counts for each time point
    # print("Maximum and minimum row counts for each time point:")
    # print(rows_per_time.max())
    # print(rows_per_time.min())
    #
    # # Select the data point that appears in '数据采集时间2023-02-06 02:00:00' but not in '数据采集时间2023-02-06 01:30:00'
    # extra_data = yuanchuan_demand[yuanchuan_demand['数据采集时间'] == '2023-02-06 02:00:00'].iloc[0]
    # print("Extra data point:")
    # print(extra_data)

    return shouchao_base_demand, yuanchuan_demand


def merge_and_process_data(all_mode_factors, shouchao_base_demand):
    # 获取手抄表用水模式

    merged_dict = {}

    # 遍历不同的类别键
    for key in all_mode_factors:
        # 检查该类别是否存在于基础用水量字典中
        if key in shouchao_base_demand:
            # 根据数据采集时间将模式因子和基础用水量进行合并
            merged_df = pd.merge(all_mode_factors[key], shouchao_base_demand[key], on='数据采集时间')

            # 计算瞬时流量：模式因子 * 每30分钟平均用水量
            merged_df['瞬时流量'] = merged_df['ModeFactor'] * merged_df['每30分钟平均用水量']
            merged_dict[key] = merged_df

    # 将 merged_dict 中的所有值合并为一个大的 DataFrame
    merged_values = list(merged_dict.values())
    merged_dataframe = pd.concat(merged_values, axis=0, ignore_index=True)

    # 从 merged_dataframe 中选择需要的列
    selected_columns = ['数据采集时间', '瞬时流量', '用户编号', 'user_junction']
    selected_merged_dataframe = merged_dataframe[selected_columns]

    # # 检查是否存在空值
    # print("每一个时刻各个手抄表用水量是否存在空值：")
    # print(selected_merged_dataframe['瞬时流量'].isnull().any())
    # # 使用 groupby 对数据采集时间进行分组，并获取每个时间点对应的行数
    # rows_per_time = selected_merged_dataframe.groupby('数据采集时间').size()
    # print(rows_per_time)
    # print(selected_merged_dataframe)

    return selected_merged_dataframe
'''