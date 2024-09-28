import cv2
import numpy as np
import os
from copy import deepcopy


def detect_bifurcation_point(edges):
    edges_nor_1 = deepcopy(edges)
    edges_nor_1 = np.pad(edges_nor_1, (1,1),mode="constant",constant_values=0)
    direct = [(0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1)]
    bifurcation_points = []
    for i in range(edges_nor_1.shape[0]):
        for j in range(edges_nor_1.shape[1]):
            if edges_nor_1[i][j] != 0:
                p = [0] * 8
                n = 0
                for k in range(len(p)):
                    p[k] = edges_nor_1[i+direct[k][0]][j+direct[k][1]]
                for k in range(len(p)):
                    if p[k%8] != 0:
                        if p[(k-1)%8] == 0 and p[(k+1)%8] == 0:
                            n += 1
                        elif p[(k+1)%8] != 0 and p[(k-1)%8] ==0 and p[(k+2)%8] == 0:
                            n +=1
                        elif p[(k+1)%8] != 0 and p[(k+2)%8] !=0 and p[(k-1)%8] == 0 and p[(k+3)%8] == 0:
                            n +=1
                        elif p[(k+1)%8] != 0 and p[(k+2)%8] !=0 and p[(k+3)%8] !=0 and p[(k-1)%8] == 0 and p[(k+4)%8] == 0:
                            n +=1
                        if n >= 3:
                            #print(n)
                            bifurcation_points.append((i,j))
    #print(bifurcation_points)
    return bifurcation_points

def connectivity_check(edges, point):
    direct = [(0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1)]
    y, x = point
    n = 0
    for i in direct:
        if (
            y + i[0] < 0
            or y + i[0] >= edges.shape[0]
            or x + i[1] < 0
            or x + i[1] >= edges.shape[1]
        ):
            continue
        elif edges[y + i[0]][x + i[1]] != 0:
            #print((y + i[0],x + i[1]))
            n = n + 1
    #print(n)
    return n


def line_check(point, edges_mark, bifurcation_points, lines, lineno):
    direct = [(0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1)]
    direct_index = 0
    y, x = point
    for i in range(len(direct)):
        if (
            y + direct[i][0] < 0
            or y + direct[i][0] >= edges_mark.shape[0]
            or x + direct[i][1] < 0
            or x + direct[i][1] >= edges_mark.shape[1]
        ):
            # direct_index += 1
            continue
        # elif edges_mark[y + direct[i][0]][x + direct[i][1]] == 0:
        # direct_index += 1
        elif edges_mark[y + direct[i][0]][x + direct[i][1]] != 0:
            direct_index = i
            edges_mark[y][x] = 0
            #print(edges_mark[y][x])
            y = y + direct[i][0]
            x = x + direct[i][1]
            #print((y,x),edges_mark[y][x])
            lines[lineno[0]].append((y, x))
            is_not_endpoint = 1

            while is_not_endpoint == 1:
                n = connectivity_check(edges_mark, (y, x))
                #print((y,x,n))
                if n == 0:
                    is_not_endpoint = 0
                    edges_mark[y][x] = 0
                elif n != 0:
                    #edges_mark[y][x] = n
                    if (edges_mark[y + direct[direct_index][0]][x + direct[direct_index][1]] != 0):
                        edges_mark[y][x] = 0
                        y = y + direct[direct_index][0]
                        x = x + direct[direct_index][1]
                        lines[lineno[0]].append((y, x))
                    elif (edges_mark[y + direct[direct_index][0]][x + direct[direct_index][1]] == 0):
                        for p in [1, 2, 3]:
                            tmp = (direct_index + p) % 8
                            y1 = y + direct[tmp][0]
                            x1 = x + direct[tmp][1]
                            if 0 <= y1 < edges_mark.shape[0] and 0 <= x1 < edges_mark.shape[1]:
                                if edges_mark[y1][x1] != 0:
                                    edges_mark[y][x] = 0
                                    direct_index = tmp
                                    y = y1
                                    x = x1
                                    lines[lineno[0]].append((y,x))
                                    break
                                elif edges_mark[y1][x1] == 0:
                                    tmp = (direct_index - p) % 8
                                    y1 = y + direct[tmp][0]
                                    x1 = x + direct[tmp][1]
                                    if edges_mark[y1][x1] != 0:
                                        edges_mark[y][x] = 0
                                        direct_index = tmp
                                        y = y1
                                        x = x1
                                        lines[lineno[0]].append((y,x))
                                        break
    point_b = []
    for j in lines[lineno[0]]:
        if j in bifurcation_points and connectivity_check(edges_mark, j) > 0:
            point_b.append(j)
    for j in point_b:
        edges_mark[j] = 1


def closed_line_check(point, edges_mark, bifurcation_points, lines, lineno):
    direct = [(0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1)]
    direct_index = 0
    y, x = point
    for i in range(len(direct)):
        if (
            y + direct[i][0] < 0
            or y + direct[i][0] >= edges_mark.shape[0]
            or x + direct[i][1] < 0
            or x + direct[i][1] >= edges_mark.shape[1]
        ):
            # direct_index += 1
            continue
        # elif edges_mark[y + direct[i][0]][x + direct[i][1]] == 0:
        # direct_index += 1
        elif edges_mark[y + direct[i][0]][x + direct[i][1]] != 0:
            direct_index = i
            edges_mark[y][x] = 0
            #print(edges_mark[y][x])
            y = y + direct[i][0]
            x = x + direct[i][1]
            #print((y,x),edges_mark[y][x])
            lines[lineno[0]].append((y, x))
            is_not_endpoint = 1

            while is_not_endpoint == 1:
                n = connectivity_check(edges_mark, (y, x))
                #print((y,x,n))
                if n == 0:
                    edges_mark[y][x] = 0
                    is_not_endpoint = 0
                elif n != 0:
                    #edges_mark[y][x] = n
                    if (edges_mark[y + direct[direct_index][0]][x + direct[direct_index][1]] != 0):
                        edges_mark[y][x] = 0
                        y = y + direct[direct_index][0]
                        x = x + direct[direct_index][1]
                        lines[lineno[0]].append((y, x))
                    elif (edges_mark[y + direct[direct_index][0]][x + direct[direct_index][1]] == 0):
                        for p in [1, 2, 3]:
                            tmp = (direct_index + p) % 8
                            y1 = y + direct[tmp][0]
                            x1 = x + direct[tmp][1]
                            if 0 <= y1 < edges_mark.shape[0] and 0 <= x1 < edges_mark.shape[1]:
                                if edges_mark[y1][x1] != 0:
                                    edges_mark[y][x] = 0
                                    direct_index = tmp
                                    y = y1
                                    x = x1
                                    lines[lineno[0]].append((y,x))
                                    break
                                elif edges_mark[y1][x1] == 0:
                                    tmp = (direct_index - p) % 8
                                    y1 = y + direct[tmp][0]
                                    x1 = x + direct[tmp][1]
                                    if edges_mark[y1][x1] != 0:
                                        edges_mark[y][x] = 0
                                        direct_index = tmp
                                        y = y1
                                        x = x1
                                        lines[lineno[0]].append((y,x))
                                        break
    point_b = []
    for j in lines[lineno[0]]:
        if j in bifurcation_points and connectivity_check(edges_mark, j) > 0:
            point_b.append(j)
    for j in point_b:
        edges_mark[j] = 1
    d = np.sqrt(np.power((lines[lineno[0]][0][0] - lines[lineno[0]][-1][0]), 2) + np.power((lines[lineno[0]][0][1] - lines[lineno[0]][-1][1]), 2))
    if d < 2 and len(lines[lineno[0]]) > 2:
        lines[lineno[0]].append(lines[lineno[0]][0])




def generate_edgepoints(edges, bifurcation_points):
    edges_mark = deepcopy(edges)
    edges_mark = np.pad(edges_mark,(1,1),mode="constant",constant_values=0)
    height, width = edges_mark.shape[:2]
    lines = []
    lineno = [0]
    is_line_present = True
    while is_line_present:
        lines_prev = deepcopy(lines)
        for i in range(height):
            for j in range(width):
                if edges_mark[i][j] != 0:
                    if connectivity_check(edges_mark, (i, j)) == 1:  #  Check if it is an endpoint
                        lines.append([])
                        lines[lineno[0]].append((i, j))
                        is_line_present = line_check((i, j), edges_mark, bifurcation_points, lines, lineno)
                        #print(lines[lineno[0]])
                        lineno[0] += 1
                        print(lineno)
                    else:
                        continue
        is_line_present = False if len(lines) == len(lines_prev) else True
    is_closed_line_present = True
    while is_closed_line_present:
        lines_prev = deepcopy(lines)
        for i in range(height):
            for j in range(width):
                if edges_mark[i][j] != 0:
                    if connectivity_check(edges_mark, (i, j)) >= 1:  # Check if it is a point on a closed edge
                        lines.append([])
                        lines[lineno[0]].append((i, j))
                        is_line_present = closed_line_check((i, j), edges_mark, bifurcation_points, lines, lineno)
                        lineno[0] += 1
                        print(lineno)
                    else:
                        continue
        is_closed_line_present = False if len(lines) == len(lines_prev) else True
    return lines


def main():
    edge = cv2.imread("src.png", 0)
    edge_nor = np.zeros_like(edge)
    for i in range(edge_nor.shape[0]):
        for j in range(edge_nor.shape[1]):
            edge_nor[i][j] = 0 if edge[i][j] == 0 else 1
    bifurcation_points = detect_bifurcation_point(edge_nor)
    lines = generate_edgepoints(edge_nor, bifurcation_points)
    if os.path.exists('linestxt') is False:
        os.mkdir('linestxt')
    elif len(os.listdir('linestxt')) > 0:
        linetxt_list = os.listdir('linestxt')
        for f in linetxt_list:
            os.unlink(os.path.join('linestxt',f))

    for i in range(len(lines)):
        line = np.empty((0,3), dtype=int)
        for j in range(len(lines[i])):
            (y,x) = [p-1 for p in lines[i][j]]
            if j == 0:
                line = np.append(line, [[x, edge_nor.shape[0] - y, 1]], 0)
            line = np.append(line, [[x, edge_nor.shape[0] - y, 0]], 0)
            if j == len(lines[i]) - 1:
                line = np.append(line, [[x, edge_nor.shape[0] - y, 2]], 0)
        np.savetxt(os.path.join('linestxt', str(i)+'.txt'), line, fmt='%.10f')

if __name__ == "__main__":
    main()
