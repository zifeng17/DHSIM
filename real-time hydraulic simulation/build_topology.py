#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/2/28 13:49
# @Author  : Yuan Wei
# @File    : build_topology.py
# @Software: PyCharm
# 批量添加节点的函数
def add_junctions(d, junction_list):
    for junction_info in junction_list:
        junction_id, elevation, x_coord, y_coord = junction_info
        junctionElevation_ft = elevation * 3.280839#默认inp文件高程是英尺单位，toolkit自动换成米（换句话说如果原始数据就是米的话就要除以0.3048再换算回来）
        index = d.addNodeJunction(junction_id)#"'" + junction_id + "'"
        #d.addPattern(junction_id)
        #d.setNodeJunctionData(index, junctionElevation_ft, 1, '')
        d.setNodeElevations(index, junctionElevation_ft)
        d.setNodeCoordinates(index, [x_coord, y_coord])
        #d.setNodeBaseDemands(index, 1/448.831000)


# 批量添加水库的函数
def add_reservoirs(d, reservoir_list):
    for reservoir_info in reservoir_list:
        reservoir_id, x_coord, y_coord = reservoir_info
        index = d.addNodeReservoir(reservoir_id)#"'" + reservoir_id + "'"
        d.setNodeCoordinates(index, [x_coord, y_coord])

# 批量添加水箱的函数
def add_tanks(d, tank_list):
    for tank_info in tank_list:
        tank_id, x_coord, y_coord = tank_info
        index = d.addNodeTank(tank_id)#"'" + tank_id + "'"
        d.setNodeCoordinates(index, [x_coord, y_coord])

# 批量添加管道的函数
def add_pipes(d, pipe_list):
    for pipe_info in pipe_list:
        pipe_id, from_node, to_node, diameter, length, roughness_coefficient = pipe_info
        index = d.addLinkPipe(pipe_id, from_node, to_node)#"'" + pipe_id + "'", "'" + from_node + "'", "'" + to_node + "'"
        diameter_in = diameter * 0.0394
        length_ft = length * 3.280839
        d.setLinkDiameter(index, diameter_in)#默认读取inp文件的节点高程是英寸单位，toolkit自动换成毫米（换句话说如果原始数据就是毫米的话就要除以25.4再换算回来）
        d.setLinkLength(index, length_ft)
        d.setLinkRoughnessCoeff(index, roughness_coefficient)

def add_valves(d, type, valve_list):
    if type == 'PRV':
        for valve_info in valve_list:
            valve_id, from_node, to_node, diameter = valve_info
            index = d.addLinkValvePRV(valve_id, from_node, to_node)
            diameter_in = diameter * 0.0394
            d.setLinkDiameter(index, diameter_in)
    if type == 'TCV':
        for valve_info in valve_list:
            valve_id, from_node, to_node, diameter = valve_info
            index = d.addLinkValveTCV(valve_id, from_node, to_node)
            diameter_in = diameter * 0.0394
            d.setLinkDiameter(index, diameter_in)




