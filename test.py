#! /usr/bin/env python

import itx_import
file = '/home/hats/gdutton/2014bld3491847.1.itx'
c = itx_import.ITX(file)
ch = 0
c.wide_spike_filter(ch, start=60)
c.display(ch)

"""
data = [1066, 1069, 1070, 1545, 1546, 1547, 1548, 1549, 1550, 1552, 1553]

def findgroups(indx):
    ''' Returns first and last point pairs in a group of indices.  '''
    
    if len(indx) < 2:
        return []
    
    indxthresh = 3     # largest gap in points
        
    indxdiff = [indx[i+1]-indx[i] for i in range(len(indx)-1)]
    
    pt1 = indx[0]
    groups = []
    for i, v in enumerate(indxdiff):
        if v > indxthresh:
            pt2 = indx[i]
            groups.append((pt1, pt2))
            pt1 = indx[i+1]
    pt2 = indx[-1]
    groups.append((pt1, pt2))
    
    return groups

print findgroups(data)
"""