import pandas as pd
import numpy as np
import sys
#第1步：正常重采样----------------------------------yw
def scada_resample_1(df_scada, start_time, end_time):
    # 将 '数据采集时间' 列转换为 datetime 对象
    df_scada['数据采集时间'] = pd.to_datetime(df_scada['数据采集时间'])
    #print(df_scada.columns)
    # 流量值全部变正
    df_scada['瞬时流量'] = pd.to_numeric(df_scada['瞬时流量'], errors='coerce')
    df_scada.loc[df_scada['瞬时流量'] < 0, '瞬时流量'] = abs(df_scada['瞬时流量'])
    df_scada['压力'] = pd.to_numeric(df_scada['压力'], errors='coerce')
    df_scada.loc[df_scada['压力'] < 0, '压力'] = abs(df_scada['压力'])

    # 删除重复值（数据采集时间和用户编号同时相同即认为重复）
    df_scada.drop_duplicates(subset=['数据采集时间', 'scada_id'], keep='first', inplace=True)

    # 根据用户编号分组（即：逐个对每个表号的数据进行处理）
    grouped = df_scada.groupby('scada_id')
    # 初始化一个空的数据框用于存储处理后的水表数据
    df_resampled = pd.DataFrame()
    #逐块表进行重采样
    for user_id, user_data in grouped:
        # print(user_data)
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
        resampled_data['scada_id'] = user_id
        # 提取关联节点

        resampled_data['junction(压力表所关联节点)'] = user_data['junction(压力表所关联节点)'].iloc[0]
        resampled_data['pipe(流量表所关联管段)'] = user_data['pipe(流量表所关联管段)'].iloc[0]
        resampled_data['是否为水库或泵站(是为1，不是为0)'] = user_data['是否为水库或泵站(是为1，不是为0)'].iloc[0]
        # 把索引列改为普通列
        resampled_data.reset_index(inplace=True)
        # 把所有数据存放在同一个数据框
        df_resampled = df_resampled.append(resampled_data)
        # 最后结果保留的列
        df_resampled = df_resampled[['数据采集时间', '瞬时流量', '压力', 'scada_id', 'junction(压力表所关联节点)',
         'pipe(流量表所关联管段)', '是否为水库或泵站(是为1，不是为0)']]
    return df_resampled

#采样时间完整化----------------------------yw
def complete_time_series(df_resampled, start_time, end_time):
    scada_complete_time_series = df_resampled.copy()
    grouped = scada_complete_time_series.groupby('scada_id')
    end_time = pd.to_datetime(end_time) - pd.Timedelta(minutes=30)  # 从结束时间减去半小时
    resampled_time_intervals = pd.date_range(start=start_time, end=end_time, freq='30T')

    # 初始化合并后的 DataFrame
    scada_complete_time_series_flow = pd.DataFrame()

    for user_id, user_data in grouped:
        if len(user_data['数据采集时间']) != 48:
            missing_time_intervals = resampled_time_intervals.difference(user_data['数据采集时间'])

            # 创建新DataFrame，包含缺失的数据
            df_resampled_complete = pd.DataFrame({
                '数据采集时间': missing_time_intervals,
                '瞬时流量': np.nan,
                '压力': np.nan,
                'scada_id': user_id,
                'junction(压力表所关联节点)': user_data['junction(压力表所关联节点)'].values[0],
                'pipe(流量表所关联管段)': user_data['pipe(流量表所关联管段)'].values[0],
                '是否为水库或泵站(是为1，不是为0)': user_data['是否为水库或泵站(是为1，不是为0)'].values[0],
            })
            # 合并数据
            merged_df = pd.concat([df_resampled_complete, scada_complete_time_series], ignore_index=True)
            scada_complete_time_series = pd.concat([scada_complete_time_series_flow, merged_df], ignore_index=True)

    return scada_complete_time_series


#第2步：流量在邻近步长进行取值-------------以790000260和780000333为例核查无误-----yw
def scada_resample_2_flow(resampled_data, df_scada):
    # 将 '数据采集时间' 列转换为 datetime 对象
    df_scada['数据采集时间'] = pd.to_datetime(df_scada['数据采集时间'])

    # 流量值全部变正
    df_scada['瞬时流量'] = pd.to_numeric(df_scada['瞬时流量'], errors='coerce')
    df_scada.loc[df_scada['瞬时流量'] < 0, '瞬时流量'] = abs(df_scada['瞬时流量'])

    # 删除重复值（数据采集时间和用户编号同时相同即认为重复）
    df_scada.drop_duplicates(subset=['数据采集时间', 'scada_id'], keep='first', inplace=True)
    # 获取瞬时流量为空的所有行
    data_missing = resampled_data[resampled_data['瞬时流量'].isnull()]

    #print("远传数据为：", df_scada)
    #output_file = "C:/Users/袁伟/Desktop/ddf_scada_2222.xlsx"#----------------正确
    #data_missing.to_excel(output_file, index=False)

    scada_resample_near_flow = resampled_data.copy()
    #判断是否有缺失值
    if data_missing.empty:
        print("第1步正常采样后，所有scada数据完整")
    else:
        print("第1步正常重采样后还有数据缺失，需要进行第2步进行就近取值")

        #把缺失数据按编号进行分组
        grouped = data_missing.groupby('scada_id')
        time_window = pd.Timedelta('30T')
        for user_id, user_data in grouped:
           # 获取原始瞬时流量数据不为空的行
            scada_data = df_scada[(df_scada['scada_id'] == user_id) &
                                               (~df_scada['瞬时流量'].isnull())][['scada_id', '数据采集时间', '瞬时流量']]
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
                user_data_near = scada_data[(scada_data['scada_id'] == user_id) &
                                        (scada_data['数据采集时间'] > start_time) &
                                        (scada_data['数据采集时间'] < end_time)]
                #print("最接近的内容：", user_data_near)
                if not user_data_near.empty:
                    # 计算时间最接近的数据点（要注意可能有多个最接近时间点，目前这里取第一个）
                    closest_time_index = user_data_near['数据采集时间'].sub(data_missing_time).abs().idxmin()
                    closest_time = user_data_near.loc[closest_time_index, '数据采集时间']

                    closest_value = user_data_near.loc[closest_time_index, '瞬时流量']

                    scada_resample_near_flow.loc[(scada_resample_near_flow['数据采集时间'] == data_missing_time) &
                                                (scada_resample_near_flow['scada_id'] == user_id), '瞬时流量'] = closest_value
                else:
                    scada_resample_near_flow.loc[(scada_resample_near_flow['数据采集时间'] == data_missing_time) &
                                                (scada_resample_near_flow['scada_id'] == user_id), '瞬时流量'] = np.nan
    return scada_resample_near_flow

#第2步：压力在邻近步长进行取值------------------yw
def scada_resample_2_pressure(resampled_data, df_scada):
    # 将 '数据采集时间' 列转换为 datetime 对象
    df_scada['数据采集时间'] = pd.to_datetime(df_scada['数据采集时间'])

    # 压力值全部变正
    df_scada['压力'] = pd.to_numeric(df_scada['压力'], errors='coerce')
    df_scada.loc[df_scada['压力'] < 0, '压力'] = abs(df_scada['压力'])

    # 删除重复值（数据采集时间和用户编号同时相同即认为重复）
    df_scada.drop_duplicates(subset=['数据采集时间', 'scada_id'], keep='first', inplace=True)
    # 获取压力值为空的所有行
    data_missing = resampled_data[resampled_data['压力'].isnull()]

    #print("远传数据为：", df_scada)
    #output_file = "C:/Users/袁伟/Desktop/ddf_scada_2222.xlsx"#----------------正确
    #data_missing.to_excel(output_file, index=False)

    scada_resample_near_pressure = resampled_data.copy()
    #判断是否有缺失值
    if data_missing.empty:
        print("第1步正常采样后，所有scada数据完整")
    else:
        print("第1步正常重采样后还有数据缺失，需要进行第2步进行就近取值")

        #把缺失数据按编号进行分组
        grouped = data_missing.groupby('scada_id')
        time_window = pd.Timedelta('30T')
        for user_id, user_data in grouped:
           # 获取原始瞬时压力数据不为空的行
            scada_data = df_scada[(df_scada['scada_id'] == user_id) &
                                               (~df_scada['压力'].isnull())][['scada_id', '数据采集时间', '压力']]
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
                user_data_near = scada_data[(scada_data['scada_id'] == user_id) &
                                        (scada_data['数据采集时间'] > start_time) &
                                        (scada_data['数据采集时间'] < end_time)]
                #print("最接近的内容：", user_data_near)
                if not user_data_near.empty:
                    # 计算时间最接近的数据点（要注意可能有多个最接近时间点，目前这里取第一个）
                    closest_time_index = user_data_near['数据采集时间'].sub(data_missing_time).abs().idxmin()
                    closest_time = user_data_near.loc[closest_time_index, '数据采集时间']

                    closest_value = user_data_near.loc[closest_time_index, '压力']

                    scada_resample_near_pressure.loc[(scada_resample_near_pressure['数据采集时间'] == data_missing_time) &
                                                (scada_resample_near_pressure['scada_id'] == user_id), '压力'] = closest_value
                else:
                    scada_resample_near_pressure.loc[(scada_resample_near_pressure['数据采集时间'] == data_missing_time) &
                                                (scada_resample_near_pressure['scada_id'] == user_id), '压力'] = np.nan
    return scada_resample_near_pressure


def linear_interpolation(time_1, time_2, value_1, value_2,target_time):
    # 计算时间差值
    time_difference = (target_time - time_1).total_seconds()
    total_time_range = (time_2 - time_1).total_seconds()

    # 进行线性插值
    interpolated_value = value_1 + (time_difference * (value_2 - value_1) / total_time_range)
    return interpolated_value

#第3步：流量线性插值---------------------------------------------yw
def scada_resample_3_flow(resampled_data, df_scada):
    df_scada['数据采集时间'] = pd.to_datetime(df_scada['数据采集时间'])
    df_scada['瞬时流量'] = pd.to_numeric(df_scada['瞬时流量'], errors='coerce')
    df_scada.loc[df_scada['瞬时流量'] < 0, '瞬时流量'] = abs(df_scada['瞬时流量'])
    df_scada.drop_duplicates(subset=['数据采集时间', 'scada_id'], keep='first', inplace=True)
    #获取进行第1、2步之后瞬时流量为空的数据
    data_missing = resampled_data[resampled_data['瞬时流量'].isnull()]

    if data_missing.empty:
        print("第2步就近取值后，scada数据已经完整")
    else:
        print("第2步就近取值后，还有数据缺失，需要进行第3步线性插值")
        grouped = data_missing.groupby('scada_id')

        for user_id, user_data in grouped:
            #获取原始瞬时流量数据不为空的行
            scada_data_no_NAN = df_scada[(df_scada['scada_id'] == user_id) &
                                                   (~df_scada['瞬时流量'].isnull())][['scada_id', '数据采集时间', '瞬时流量']]

            data_missing_times = user_data['数据采集时间']

            for data_missing_time in data_missing_times:
                closest_times = scada_data_no_NAN .loc[scada_data_no_NAN ['scada_id'] == user_id, '数据采集时间']\
                    .sort_values(key=lambda x: abs(x - data_missing_time)).head(2)
                #判断邻近时间点是否有2个（保证时间点和流量值都存在）
                #print(user_id)
                if len(closest_times) == 2:
                    time_1 = closest_times.iloc[0]  # 获取第一个时间点
                    time_2 = closest_times.iloc[1]  # 获取第二个时间点
                    #print("插值时间1为：",time_1)
                    #print("插值时间2为：",time_2)
                    # 把数据采集时间设置为索引
                    scada_data_no_NAN .set_index('数据采集时间', inplace=True)
                    value_1 = scada_data_no_NAN .loc[time_1, '瞬时流量']
                    value_2 = scada_data_no_NAN .loc[time_2, '瞬时流量']

                    #print("数值1为：",value_1)
                    #print("数值2为：",value_2)

                    # 调用线性插值函数并输出结果
                    desire_value = linear_interpolation(time_1, time_2, value_1, value_2, data_missing_time)
                    resampled_data.loc[(resampled_data['数据采集时间'] == data_missing_time) &
                                        (resampled_data['scada_id'] == user_id), '瞬时流量'] = desire_value
                    #把数据采集时间恢复普通列（不然前面closest_times处会出问题）
                    scada_data_no_NAN .reset_index(inplace=True)
                else:
                    print("时间点少于2个，无法插值")
                    resampled_data.loc[(resampled_data['数据采集时间'] == data_missing_time) &
                                       (resampled_data['scada_id'] == user_id), '瞬时流量'] = np.nan
    return resampled_data


#第3步：压力线性插值-------------------------------yw
def scada_resample_3_pressure(resampled_data, df_scada):
    df_scada['数据采集时间'] = pd.to_datetime(df_scada['数据采集时间'])
    df_scada['压力'] = pd.to_numeric(df_scada['压力'], errors='coerce')
    df_scada.loc[df_scada['压力'] < 0, '压力'] = abs(df_scada['压力'])
    df_scada.drop_duplicates(subset=['数据采集时间', 'scada_id'], keep='first', inplace=True)
    #获取进行第1、2步之后压力为空的数据
    data_missing = resampled_data[resampled_data['压力'].isnull()]

    if data_missing.empty:
        print("第2步就近取值后，scada数据已经完整")
    else:
        print("第2步就近取值后，还有数据缺失，需要进行第3步线性插值")
        grouped = data_missing.groupby('scada_id')

        for user_id, user_data in grouped:
            #获取原始压力数据不为空的行
            scada_data_no_NAN = df_scada[(df_scada['scada_id'] == user_id) &
                                                   (~df_scada['压力'].isnull())][['scada_id', '数据采集时间', '压力']]

            data_missing_times = user_data['数据采集时间']

            for data_missing_time in data_missing_times:
                closest_times = scada_data_no_NAN .loc[scada_data_no_NAN ['scada_id'] == user_id, '数据采集时间']\
                    .sort_values(key=lambda x: abs(x - data_missing_time)).head(2)
                #判断邻近时间点是否有2个（保证时间点和压力值都存在）
                #print(user_id)
                if len(closest_times) == 2:
                    time_1 = closest_times.iloc[0]  # 获取第一个时间点
                    time_2 = closest_times.iloc[1]  # 获取第二个时间点
                    #print("插值时间1为：",time_1)
                    #print("插值时间2为：",time_2)
                    # 把数据采集时间设置为索引
                    scada_data_no_NAN .set_index('数据采集时间', inplace=True)
                    value_1 = scada_data_no_NAN .loc[time_1, '压力']
                    value_2 = scada_data_no_NAN .loc[time_2, '压力']

                    #print("数值1为：",value_1)
                    #print("数值2为：",value_2)

                    # 调用线性插值函数并输出结果
                    desire_value = linear_interpolation(time_1, time_2, value_1, value_2, data_missing_time)
                    resampled_data.loc[(resampled_data['数据采集时间'] == data_missing_time) &
                                        (resampled_data['scada_id'] == user_id), '压力'] = desire_value
                    #把数据采集时间恢复普通列（不然前面closest_times处会出问题）
                    scada_data_no_NAN .reset_index(inplace=True)
                else:
                    print("时间点少于2个，无法插值")
                    resampled_data.loc[(resampled_data['数据采集时间'] == data_missing_time) &
                                       (resampled_data['scada_id'] == user_id), '压力'] = np.nan
    return resampled_data


#第4步：瞬时流量为空的置0-------------------------------------------yw
def scada_resample_4_flow(resampled_data):
    #因为线性插值条件较松，插值后可能出现负值
    resampled_data['数据采集时间'] = pd.to_datetime(resampled_data['数据采集时间'])
    resampled_data['瞬时流量'] = pd.to_numeric(resampled_data['瞬时流量'], errors='coerce')
    resampled_data.loc[resampled_data['瞬时流量'] < 0, '瞬时流量'] = abs(resampled_data['瞬时流量'])
    resampled_data.drop_duplicates(subset=['数据采集时间', 'scada_id'], keep='first', inplace=True)

    data_missing = resampled_data[resampled_data['瞬时流量'].isnull()]
    # 判断是否有缺失值
    if data_missing.empty:
        print("第3步线性插值后，所有远传数据完整")
    else:
        print("第3步线性插值后，还有数据缺失（有数据不能插值）")
        grouped = data_missing.groupby('scada_id')
        for user_id, user_data in grouped:
            row_count = len(user_data)
            if row_count < 48:#每个id对应的空行少于48意味着一天中至少有一个数，直接把剩余的空值置为0
                data_missing_times = user_data['数据采集时间']
                for data_missing_time in data_missing_times:
                    resampled_data.loc[(resampled_data['数据采集时间'] == data_missing_time) &
                                    (resampled_data['scada_id'] == user_id), '瞬时流量'] = 0
            else:
                print("scada表一天无数据或者")
    return resampled_data
#第4步：压力为空的置0-------------------------------------------yw
def scada_resample_4_pressure(resampled_data):
    #因为线性插值条件较松，插值后可能出现负值
    resampled_data['数据采集时间'] = pd.to_datetime(resampled_data['数据采集时间'])
    resampled_data['压力'] = pd.to_numeric(resampled_data['压力'], errors='coerce')
    resampled_data.loc[resampled_data['压力'] < 0, '压力'] = abs(resampled_data['压力'])
    resampled_data.drop_duplicates(subset=['数据采集时间', 'scada_id'], keep='first', inplace=True)

    data_missing = resampled_data[resampled_data['压力'].isnull()]
    # 判断是否有缺失值
    if data_missing.empty:
        print("第3步线性插值后，所有远传数据完整")
    else:
        print("第3步线性插值后，还有数据缺失（有数据不能插值）")
        grouped = data_missing.groupby('scada_id')
        for user_id, user_data in grouped:
            row_count = len(user_data)
            if row_count < 48:#每个id对应的空行少于48意味着一天中至少有一个数，直接把剩余的空值置为0
                data_missing_times = user_data['数据采集时间']
                for data_missing_time in data_missing_times:
                    resampled_data.loc[(resampled_data['数据采集时间'] == data_missing_time) &
                                    (resampled_data['scada_id'] == user_id), '压力'] = 0
            else:
                print("scada表一天无数据或者")
    return resampled_data


#判断所有时间序列数据是否齐全（逐块表进行核查）---------------------------------------yw
def complete_time_series_verification_flow(resampled_data):
    #不包含瞬时流量为空的行
    resampled_data_no_nan = resampled_data[~pd.isna(resampled_data['瞬时流量'])]
    grouped = resampled_data_no_nan.groupby('scada_id')
    for user_id, user_data in grouped:
        # 判断每块表数据是否为48
        if len(user_data['瞬时流量']) != 48:
            print("缺失的用户编号：", user_id)
            raise ValueError("scada数据缺失")

    #把流量单位换成m3/s
    #resampled_data['瞬时流量'] = resampled_data['瞬时流量']/3600
    # 把流量单位换成GPM
    resampled_data['瞬时流量'] = resampled_data['瞬时流量'] * 4.403
    print("经过数据处理后，所有scada流量数据均完整")
    return resampled_data

#判断所有时间序列数据是否齐全（逐块表进行核查）---------------------------------------yw
def complete_time_series_verification_pressure(resampled_data):
    #不包含瞬时流量为空的行
    resampled_data_no_nan = resampled_data[~pd.isna(resampled_data['压力'])]
    grouped = resampled_data_no_nan.groupby('scada_id')
    for user_id, user_data in grouped:
        # 判断每块表数据是否为48
        if len(user_data['压力']) != 48:
            print("缺失的用户编号：", user_id)
            raise ValueError("scada数据缺失")

    #把压力单位换算成m
    #resampled_data['压力'] = resampled_data['压力']*101.477
    #把压力单位换算成psi
    resampled_data['压力'] = resampled_data['压力'] * 145
    print("经过数据处理后，所有scada压力数据均完整")
    return resampled_data

#把压力和流量数据合并--------------------------yw
def merge_flow_and_pressure(scada_resample_set_NAN_flow, scada_resample_set_NAN_pressure):
    scada_resample_set_NAN_flow['压力'] = scada_resample_set_NAN_pressure['压力']
    return scada_resample_set_NAN_flow


#设置边界条件------------------------沿用LHF
def get_bian_jie_tiao_jian_data(merged_data):
    #bian_jie_tiao_jian_data = merged_data[merged_data['是否为水库或泵站(是为1，不是为0)'] == 1]
    bian_jie_tiao_jian_data = merged_data[(merged_data['是否为水库或泵站(是为1，不是为0)'] == 1) & (merged_data['压力'].notna())]
    #print("边界条件：", bian_jie_tiao_jian_data.head(300).to_string())

    # 根据 junction 进行聚类
    grouped_pres_clusters = bian_jie_tiao_jian_data.groupby('junction(压力表所关联节点)')
    # 创建一个字典来存储聚类结果
    clustering_results_pres = {}
    # 遍历每个聚类并获取时间序列数据
    for junction, group in grouped_pres_clusters:
        clustering_results_pres[junction] = group[['数据采集时间', '压力']].copy()

    grouped_clustering_results_of_bianjie_pres = {}
    for junction, data in clustering_results_pres.items():
        for index, row in data.iterrows():
            timestamp = row['数据采集时间']
            if timestamp not in grouped_clustering_results_of_bianjie_pres:
                grouped_clustering_results_of_bianjie_pres[timestamp] = {}
            if junction not in grouped_clustering_results_of_bianjie_pres[timestamp]:
                grouped_clustering_results_of_bianjie_pres[timestamp][junction] = 0
            grouped_clustering_results_of_bianjie_pres[timestamp][junction] += row['压力']
    # print(grouped_clustering_results_pres)

    return grouped_clustering_results_of_bianjie_pres


#非边界压力监测点的压力监测值和关联的节点ID（整理监测值的格式，便于与模拟值比较）-----沿用LHF
def pressure_scada(merged_data):
    # print(f'merged_data\n{merged_data}')
    #获取全部压力表数据
    #scada_pressure_data = merged_data[(merged_data['是否为水库或泵站(是为1，不是为0)'] == 0)]
    #清除压力和关联节点列为空的行
    scada_pressure_data = merged_data[(merged_data['junction(压力表所关联节点)'].notna()) & (merged_data['压力'].notna())]

    #print("scada_pressure_data:", scada_pressure_data)

    # 删除junction(压力所在节点)列所在行为空的行
    #scada_pressure_data = scada_pressure_data.dropna(subset=['压力'])
    # print('jiedianyali\n', scada_pressure_data.head(10).to_string())

    # 根据 junction 进行聚类
    grouped_pres_clusters = scada_pressure_data.groupby('junction(压力表所关联节点)')
    # 创建一个字典来存储聚类结果
    clustering_results_pres = {}
    # 遍历每个聚类并获取时间序列数据
    for junction, group in grouped_pres_clusters:
        if junction.strip() != "":  # 检查键是否为空
            clustering_results_pres[junction] = group[['数据采集时间', '压力']].copy()

    #print("clustering_results_pres:",clustering_results_pres)
    # 获取所有值的长度
    value_lengths = [len(data) for data in clustering_results_pres.values()]
    # 找到值的长度是否都相同
    is_same_length = all(length == value_lengths[0] for length in value_lengths)
    # 如果值的长度不相同，则删除对应的键值对
    if not is_same_length:
        clustering_results_pres = {k: v for k, v in clustering_results_pres.items() if len(v) == value_lengths[0]}
    # 删除键为空的键值对
    clustering_results_pres = {k: v for k, v in clustering_results_pres.items() if k.strip() != ""}
    # 获取clustering_results_pres的键存为一个列表
    scada_pressure_junction_list = list(clustering_results_pres.keys())
    print('-------------------------------------------------------')
    print('len(scada_pressure_junction_list):', len(scada_pressure_junction_list))
    print('-------------------------------------------------------')
    # # 打印删除后的聚类结果
    # for junction, data in clustering_results_pres.items():
    #     print(f"Data for scada_pressure_junction {junction}:")
    #     print(data, data.shape)
    #     print()

    grouped_clustering_results_of_scada_pres = {}
    for junction, data in clustering_results_pres.items():
        for index, row in data.iterrows():
            timestamp = row['数据采集时间']
            if timestamp not in grouped_clustering_results_of_scada_pres:
                grouped_clustering_results_of_scada_pres[timestamp] = {}
            if junction not in grouped_clustering_results_of_scada_pres[timestamp]:
                grouped_clustering_results_of_scada_pres[timestamp][junction] = 0
            grouped_clustering_results_of_scada_pres[timestamp][junction] += row['压力']
    # print(grouped_clustering_results_pres)

    return grouped_clustering_results_of_scada_pres, scada_pressure_junction_list

#SCADA流量表监测值/SCADA流量表关联管道ID/新基泵站SCADA流量表监测值/新基泵站SCADA流量表关联管道ID-（整理监测值的格式，便于与模拟值比较）--沿用LHF
def flow_scada(merged_data):
    #scada_flow_data = merged_data[merged_data['是否为水库或泵站(是为1，不是为0)'] == 0]

    # 删除junction(压力所在节点)列所在行为空的行
    scada_flow_data = merged_data.dropna(subset=['瞬时流量'])
    # print('jiedianflow\n', scada_flow_data.head(10).to_string())

    # 根据 pipe 进行聚类
    grouped_flow_clusters = scada_flow_data.groupby('pipe(流量表所关联管段)')
    # 创建一个字典来存储聚类结果
    clustering_results_flow = {}
    # 遍历每个聚类并获取时间序列数据
    for pipe, group in grouped_flow_clusters:
        clustering_results_flow[pipe] = group[['数据采集时间', '瞬时流量']].copy()

    # 根据 id 进行聚类--------------------------yw2024.1.6
    grouped_flow_clusters_id = scada_flow_data.groupby('scada_id')
    #创建一个字典来存储聚类结果
    clustering_results_flow_xunibiao = {}
    # 遍历每个聚类并获取时间序列数据
    for scada_id, group in grouped_flow_clusters_id:
        clustering_results_flow_xunibiao[scada_id] = group[['数据采集时间', '瞬时流量']].copy()

    # 获取所有clustering_results_flow值的长度
    value_lengths = [len(data) for data in clustering_results_flow.values()]
    # 找到值的长度是否都相同
    is_same_length = all(length == value_lengths[0] for length in value_lengths)
    # 如果值的长度不相同，则删除对应的键值对
    if not is_same_length:
        clustering_results_flow = {k: v for k, v in clustering_results_flow.items() if len(v) == value_lengths[0]}
    # 删除键为空的键值对
    clustering_results_flow = {k: v for k, v in clustering_results_flow.items() if k.strip() != ""}
    # 获取clustering_results_pres的键存为一个列表
    scada_flow_pipe_list = list(clustering_results_flow.keys())
    print('-------------------------------------------------------')
    print('len(scada_flow_pipe_list):', len(scada_flow_pipe_list))
    print('-------------------------------------------------------')
    # # 打印删除后的聚类结果
    # for pipe, data in clustering_results_flow.items():
    #     print(f"Data for scada_flow_pipe {pipe}:")
    #     print(data, data.shape)
    #     print()

    grouped_clustering_results_of_scada_flow = {}
    for junction, data in clustering_results_flow.items():
        for index, row in data.iterrows():
            timestamp = row['数据采集时间']
            if timestamp not in grouped_clustering_results_of_scada_flow:
                grouped_clustering_results_of_scada_flow[timestamp] = {}
            if junction not in grouped_clustering_results_of_scada_flow[timestamp]:
                grouped_clustering_results_of_scada_flow[timestamp][junction] = 0
            grouped_clustering_results_of_scada_flow[timestamp][junction] += row['瞬时流量']
    # print(grouped_clustering_results_of_scada_flow)

    scada_flow_data_input = merged_data[merged_data['是否为水库或泵站(是为1，不是为0)'] == 1]
    scada_flow_data = scada_flow_data_input.dropna(subset=['瞬时流量'])
    # print('jiedianflow\n', scada_flow_data.head(10).to_string())

    grouped_flow_clusters = scada_flow_data.groupby('pipe(流量表所关联管段)')
    clustering_results_flow = {}
    for pipe, group in grouped_flow_clusters:
        clustering_results_flow[pipe] = group[['数据采集时间', '瞬时流量']].copy()

    value_lengths = [len(data) for data in clustering_results_flow.values()]
    is_same_length = all(length == value_lengths[0] for length in value_lengths)

    if not is_same_length:
        clustering_results_flow = {k: v for k, v in clustering_results_flow.items() if len(v) == value_lengths[0]}

    clustering_results_flow = {k: v for k, v in clustering_results_flow.items() if k.strip() != ""}
    scada_flow_pipe_list_input = list(clustering_results_flow.keys())

    # for pipe, data in clustering_results_flow.items():
    #     print(f"Data for scada_flow_pipe_input {pipe}:")
    #     print(data, data.shape)
    #     print()

    grouped_clustering_results_of_scada_flow_input = {}
    for junction, data in clustering_results_flow.items():
        for index, row in data.iterrows():
            timestamp = row['数据采集时间']
            if timestamp not in grouped_clustering_results_of_scada_flow_input:
                grouped_clustering_results_of_scada_flow_input[timestamp] = {}
            if junction not in grouped_clustering_results_of_scada_flow_input[timestamp]:
                grouped_clustering_results_of_scada_flow_input[timestamp][junction] = 0
            grouped_clustering_results_of_scada_flow_input[timestamp][junction] += row['瞬时流量']

    grouped_clustering_results_of_scada_flow_xunibiao = {}#--------------------------yw
    for scada_id, data in clustering_results_flow_xunibiao.items():
        for index, row in data.iterrows():
            timestamp = row['数据采集时间']
            if timestamp not in grouped_clustering_results_of_scada_flow_xunibiao:
                grouped_clustering_results_of_scada_flow_xunibiao[timestamp] = {}
            if scada_id not in grouped_clustering_results_of_scada_flow_xunibiao[timestamp]:
                grouped_clustering_results_of_scada_flow_xunibiao[timestamp][scada_id] = 0
            grouped_clustering_results_of_scada_flow_xunibiao[timestamp][scada_id] += row['瞬时流量']

    return (grouped_clustering_results_of_scada_flow,
            scada_flow_pipe_list,
            grouped_clustering_results_of_scada_flow_input,
            scada_flow_pipe_list_input,
            grouped_clustering_results_of_scada_flow_xunibiao)
