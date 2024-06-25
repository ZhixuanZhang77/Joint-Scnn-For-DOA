#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   snn_heatmap.py    

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/5/31 17:00   lintean      1.0         None
'''

import numpy as np
import matplotlib.pyplot as plt

def snn_heatmap(spike,l,model):
    """
    snn的可视化 'Visual Explanations from Spiking Neural Networks using Interspike Intervals'
    @param spike: array(10, 32, 15, 15)
    """

    if len(spike.shape) < 3:
        print(spike.shape)
        print(spike)
        print("error! check the code! ")
    else:
        # SAM实现
        spike = spike.cpu().detach().numpy()
        gamma = 0.2
        __time, __channel, __height = spike.shape #, __width

        # for t in range(__time):
        #     for c in range(__channel):
        #         if np.sum(spike[t, c]) > 0:
        #             print("[" + str(t) + ", " + str(c) + "] have spike!")
        sam_spike_matrix = np.zeros(shape=[__time,__height])
        for time_step in range(__time):
            # sam_matrix是可视化的矩阵，可视化就是计算出sam_matrix
            sam_matrix = np.zeros(shape=[__height])#, __width
            print('time_step:', time_step)
            for h in range(__height):
                    for c in range(__channel):
                        ncs = 0
                        for t in range(time_step):
                            if spike[t, c, h] != 0:
                                ncs += np.exp(-gamma * (time_step - t))
                        sam_matrix[h] += ncs * spike[time_step, c, h]
            sam_spike_matrix[time_step] = sam_matrix
        # 下面那个函数其实有点难以在你那里运行，因为这个函数依赖于我这边的路径结构；
        # 其实那个函数的作用就是把sam_matrix做成热力图，所以我这里用plt.matshow直接显示出来，按回车显示下一张图片
        # plt.matshow(sam_spike_matrix)
        #
        # plt.tight_layout()
        # plt.show()
        # plt.savefig('heatmap/{0}_layer{1}.png'.format(model,l+1))

        plt.matshow(sam_spike_matrix)
        plt.colorbar()
        plt.figure(figsize=(3, 3))

        # 保存图片
        # fig = plt.gcf()
        # plt.margins(0, 0)
        # plt.savefig('heatmap/{0}_layer{1}.png'.format(model,l+1), dpi=500, bbox_inches='tight')  # dpi越高越清晰
        plt.show()

        # input()

        # 这是一个自定义可视化矩阵的函数
        # heatmap(_, log_path, window_index, epoch)
