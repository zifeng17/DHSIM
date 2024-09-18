import psycopg2
import pandas as pd
import datetime
from datetime import timedelta
class PostgreSQL_connection:
    def __init__(self, host, port, username, password, database, start_time, end_time):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.database = database
        self.start_time = start_time
        self.end_time = end_time
        self.connection = self.connect_to_PostgreSQL()

    def connect_to_PostgreSQL(self):
        try:
            connection = psycopg2.connect(
                host=self.host,
                port=self.port,
                user=self.username,
                password=self.password,
                database=self.database
            )
            print("已连接到PostgreSQL数据库")
            return connection
        except psycopg2.Error as e:
            print(f"数据库连接错误：{e}")
            return None

    def query_junction_ShiJiao(self):
        db_connection = self.connect_to_PostgreSQL()

        if db_connection:

            query = f"""SELECT * from junction_ShiJiao """
            try:
                result_df_node = pd.read_sql(query, db_connection)
                db_connection.close()
                return result_df_node

            except psycopg2.Error as e:
                print(f"junction查询出错: {e}")
                db_connection.close()
                return None

    def query_pipe_ShiJiao(self):
        db_connection = self.connect_to_PostgreSQL()

        if db_connection:

            query = f"""SELECT * from pipe_ShiJiao """
            try:
                result_df_pipe = pd.read_sql(query, db_connection)
                db_connection.close()
                return result_df_pipe

            except psycopg2.Error as e:
                print(f"pipe查询出错: {e}")
                db_connection.close()
                return None

    def query_valve_ShiJiao(self):
        db_connection = self.connect_to_PostgreSQL()

        if db_connection:

            query = f"""SELECT * from valve_ShiJiao """
            try:
                result_df_reservoir = pd.read_sql(query, db_connection)
                db_connection.close()
                return result_df_reservoir

            except psycopg2.Error as e:
                print(f"valve查询出错: {e}")
                db_connection.close()
                return None

    def query_reservoir_ShiJiao(self):
        db_connection = self.connect_to_PostgreSQL()

        if db_connection:

            query = f"""SELECT * from reservoir_ShiJiao """
            try:
                result_df_reservoir = pd.read_sql(query, db_connection)
                db_connection.close()
                return result_df_reservoir

            except psycopg2.Error as e:
                print(f"reservoir查询出错: {e}")
                db_connection.close()
                return None

    def query_tank_ShiJiao(self):
        db_connection = self.connect_to_PostgreSQL()

        if db_connection:

            query = f"""SELECT * from tank_ShiJiao """
            try:
                result_df_tank = pd.read_sql(query, db_connection)
                db_connection.close()
                return result_df_tank

            except psycopg2.Error as e:
                print(f"水箱查询出错: {e}")
                db_connection.close()
                return None

    def query_shouchao_data_demand(self, start_time):
        db_connection = self.connect_to_PostgreSQL()

        if db_connection:

            start_time = datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M')
            adjusted_start_time = start_time - timedelta(minutes=60*24*30*3)  # 查询起始时间为3个月之前


            query = f"""SELECT sudi."本次抄表时间", sudi."上次抄表时间", sudi."抄表水量", sudi."用户编号", ubit."user_junction", 
                        ubit."cus_type_code", ubit."三级分区"
                        FROM shouchao_u_detail_info sudi
                        JOIN user_base_info_type_xinjihou_an_daifenpei ubit ON sudi."用户编号" = ubit."user_id"
                        WHERE sudi."上次抄表时间" BETWEEN '{adjusted_start_time}' AND '{start_time}'
                        AND ubit."一级分区" = 'ShiJiao'
                        AND ubit."末端用户表为-1，未分配区域为1" = 1
                        AND ubit."远传表为1、手抄表为0、虚拟表为-1" = 0;
                        """
            try:
                result_df_shouchao = pd.read_sql(query, db_connection)
                db_connection.close()
                return result_df_shouchao

            except psycopg2.Error as e:
                print(f"手抄表查询出错: {e}")
                db_connection.close()
                return None

    def query_yuanchuan_data_demand(self, start_time, end_time):
        # 获取查询日期的远传表模式数据
        db_connection = self.connect_to_PostgreSQL()

        if db_connection:
            # 调整查询起止时间为重采样做准备
            start_time = datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M')
            end_time = datetime.datetime.strptime(end_time, '%Y-%m-%d %H:%M')
            adjusted_start_time = start_time - timedelta(minutes=60)  # 调整查询起始时间为半小时之前
            adjusted_end_time = end_time + timedelta(minutes=60)  # 调整查询结束时间为半小时之前
            # 构建查询语句
            query = f"""SELECT yudi."数据采集时间", yudi."瞬时流量", yudi."用户编号", ubit."user_junction", ubit."cus_type_code", ubit."三级分区", ubit."末端用户表为-1，未分配区域为1"
                        FROM yuanchuan_u_detail_info yudi
                        JOIN user_base_info_type_xinjihou_an_daifenpei ubit ON yudi."用户编号" = ubit."user_id"
                        WHERE yudi."数据采集时间" >= '{adjusted_start_time}' AND yudi."数据采集时间" < '{adjusted_end_time}'
                        AND ubit."远传表为1、手抄表为0、虚拟表为-1" = 1;
                        """
            try:
                result_df_yuanchuan_demand = pd.read_sql(query, db_connection)
                db_connection.close()
                return result_df_yuanchuan_demand

            except psycopg2.Error as e:
                print(f"远传表查询出错: {e}")
                db_connection.close()
                return None

    def query_scada_data_u(self, start_time, end_time):
        # 获取查询日期的scada_u模式数据
        db_connection = self.connect_to_PostgreSQL()

        if db_connection:
            # 调整查询起止时间为重采样做准备
            start_time = datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M')
            end_time = datetime.datetime.strptime(end_time, '%Y-%m-%d %H:%M')
            adjusted_start_time = start_time - timedelta(minutes=60)  # 调整查询起始时间为半小时之前
            adjusted_end_time = end_time + timedelta(minutes=30)  # 调整查询结束时间为半小时之后
            # 构建查询语句
            query = f"""SELECT sbin."是否为水库或泵站(是为1，不是为0)", sbin."junction(压力表所关联节点)", sbin."pipe(流量表所关联管段)",
                        sbin."scada_id", s用."数据采集时间", s用."瞬时流量", s用."压力"
                        FROM scada_base_info sbin
                        JOIN scada_u_info s用 ON sbin."scada_id" = s用."用户编号" 
                        WHERE sbin."所属区域" = 'ShiJiao' OR sbin."所属区域" = 'ChengNan/ShiJiao'
                        AND s用."数据采集时间" >= '{adjusted_start_time}' AND s用."数据采集时间" < '{adjusted_end_time}';"""

                # 从数据库查询数据
            try:
                result_df_scada= pd.read_sql(query, db_connection)
                db_connection.close()
                return result_df_scada

            except psycopg2.Error as e:
                print(f"scada_u监测表查询出错: {e}")
                db_connection.close()
                return None
    
    def query_scada_data_m(self, start_time, end_time):
        # 获取查询日期的scada_u模式数据
        db_connection = self.connect_to_PostgreSQL()

        if db_connection:
            # 调整查询起止时间为重采样做准备
            start_time = datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M')
            end_time = datetime.datetime.strptime(end_time, '%Y-%m-%d %H:%M')
            adjusted_start_time = start_time - timedelta(minutes=60)  # 调整查询起始时间为半小时之前
            adjusted_end_time = end_time + timedelta(minutes=30)  # 调整查询结束时间为半小时之后
            # 构建查询语句
            query = f"""SELECT sbin."是否为水库或泵站(是为1，不是为0)", sbin."junction(压力表所关联节点)", sbin."pipe(流量表所关联管段)",
                        sbin."scada_id", s设."数据采集时间", s设."瞬时流量", s设."压力"
                        FROM scada_base_info sbin
                        JOIN scada_m_info s设 ON sbin."scada_id" = s设."设备编号"
                        WHERE sbin."所属区域" = 'ShiJiao' OR sbin."所属区域" = 'ChengNan/ShiJiao'
                        AND s设."数据采集时间" >= '{adjusted_start_time}' AND s设."数据采集时间" < '{adjusted_end_time}';"""

                # 从数据库查询数据
            try:
                result_df_scada= pd.read_sql(query, db_connection)
                db_connection.close()
                return result_df_scada

            except psycopg2.Error as e:
                print(f"scada_m监测表查询出错: {e}")
                db_connection.close()
                return None        

    #查询虚拟表
    def query_XuNiBiao(self):
        db_connection = self.connect_to_PostgreSQL()

        if db_connection:

            query = f"""SELECT ubit."user_id", ubit."user_junction"
                        FROM user_base_info_type_xinjihou_an_daifenpei ubit
                        WHERE ubit."远传表为1、手抄表为0、虚拟表为-1" = -1;
                    
                    """
            try:
                result_df_shouchao = pd.read_sql(query, db_connection)
                db_connection.close()
                return result_df_shouchao

            except psycopg2.Error as e:
                print(f"手抄表查询出错: {e}")
                db_connection.close()
                return None

    #查询待分配区域进、出口SCADA流量表
    def query_SCADA_flow_unallocated_area(self):
        db_connection = self.connect_to_PostgreSQL()

        if db_connection:

            query = f"""SELECT dfsdi."一级分区",  dfsdi."待分配区域", dfsdi."SCADA_id", dfsdi."access_code(进水为1，出水为-1)"
                        FROM dma_flow_scada_define_info_xinjihou_an_daifenpei dfsdi
                        WHERE dfsdi."一级分区" = 'ShiJiao';

                    """
            try:
                SCADA_flow_unallocated_area_data = pd.read_sql(query, db_connection)
                db_connection.close()
                return SCADA_flow_unallocated_area_data

            except psycopg2.Error as e:
                print(f"待分配区域SCADA流量表: {e}")
                db_connection.close()
                return None

    # 查询待分配区域进、出口SCADA流量表
    def query_junction_unallocated_area(self):
        db_connection = self.connect_to_PostgreSQL()

        if db_connection:

            query = f"""SELECT jua."junction", jua."area"
                            FROM junction_unallocated_area_xinjihou_an_daifenpei jua
                    """
            try:
                unallocated_area_junction = pd.read_sql(query, db_connection)
                db_connection.close()
                return unallocated_area_junction

            except psycopg2.Error as e:
                print(f"待分配区域节点: {e}")
                db_connection.close()
                return None