#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/2/29 15:05
# @Author  : Yuan Wei
# @File    : set_controls.py
# @Software: PyCharm
from epyt import epanet
#设置断管
def set_pipe_control(d,id_list):
    for id in id_list:
        pipe_index = d.getLinkIndex(id)
        d.setLinkInitialStatus(pipe_index, 0)
