import numpy
import pandas


#
def ARC(inp_data1, inp_data2):
    try:
        assert type(inp_data1) == type(inp_data2)
        assert type(inp_data1) == numpy.ndarray or type(inp_data1) == pandas.core.frame.DataFrame or type(
            inp_data1) == pandas.core.series.Series or type(inp_data1) == list

        assert type(inp_data2) == numpy.ndarray or type(inp_data2) == pandas.core.frame.DataFrame or type(
            inp_data2) == pandas.core.series.Series or type(inp_data2) == list
        if type(inp_data1) is not list:
            assert len(inp_data1.shape) == 1 and len(inp_data2.shape) == 1
        assert len(inp_data1) == len(inp_data2)
        arc = []
        for i in range(2, len(inp_data1)):
            if type(inp_data1) == numpy.ndarray or type(inp_data1) == list:
                arc.append(numpy.corrcoef(inp_data1[0:i], inp_data2[0:i])[0][1])
            elif type(inp_data1) == pandas.core.series.Series:
                arc.append(numpy.corrcoef(inp_data1.iloc[0:i], inp_data2.iloc[0:i])[0][1])
            else:
                break
        return numpy.array(arc)
    except Exception as e:
        print(e.__traceback__.tb_lineno, e)


#
def cor_during_time(inp_data1, inp_data2, time_period):
    try:
        assert type(inp_data1) == type(inp_data2)
        assert type(inp_data1) == numpy.ndarray or type(inp_data1) == pandas.core.frame.DataFrame or type(
            inp_data1) == pandas.core.series.Series or type(inp_data1) == list

        assert type(inp_data2) == numpy.ndarray or type(inp_data2) == pandas.core.frame.DataFrame or type(
            inp_data2) == pandas.core.series.Series or type(inp_data2) == list
        if type(inp_data1) is not list:
            assert len(inp_data1.shape) == 1 and len(inp_data2.shape) == 1
        assert len(inp_data1) == len(inp_data2)
        c_d_t = []
        for i in range(len(inp_data1) - time_period):
            if type(inp_data1) == numpy.ndarray or type(inp_data1) == list:
                c_d_t.append(numpy.corrcoef(inp_data1[i:i + time_period], inp_data2[i:i + time_period])[0][1])
            elif type(inp_data1) == pandas.core.series.Series:
                c_d_t.append(numpy.corrcoef(inp_data1.iloc[i:i + time_period], inp_data2.iloc[i:i + time_period])[0][1])
            else:
                break
        return numpy.array(c_d_t)
    except Exception as e:
        print(e.__traceback__.tb_lineno, e)


#
def intersect(inp_data1, inp_data2):
    try:
        assert type(inp_data1) == type(inp_data2)
        assert type(inp_data1) == numpy.ndarray or type(inp_data1) == pandas.core.frame.DataFrame or type(
            inp_data1) == pandas.core.series.Series or type(inp_data1) == list or type(
            inp_data1) == pandas.core.indexes.range.RangeIndex or type(inp_data1) == pandas.core.indexes.base.Index
        if type(inp_data1) is not list:
            assert len(inp_data1.shape) == 1 and len(inp_data2.shape) == 1
        sect = []
        if len(inp_data1) > len(inp_data2):
            print(1)
            for i in inp_data1:
                if i in inp_data2:
                    sect.append(i)

        else:
            print(2)
            for i in inp_data2:
                if i in inp_data1:
                    sect.append(i)

        return numpy.array(sect)
    except Exception as e:
        print(e.__traceback__.tb_lineno, e)


#

def exp_change(inp_columns):
    try:
        assert type(inp_columns) == pandas.core.frame.DataFrame or type(
            inp_columns) == pandas.core.series.Series or type(
            inp_columns) == numpy.ndarray or type(inp_columns) == list
        # inp_columns2 = inp_columns[:]
        if type(inp_columns) == pandas.core.frame.DataFrame or type(inp_columns) == pandas.core.series.Series:
            inp_columns2 = inp_columns.values
        else:
            inp_columns2 = inp_columns[:]
        exp_change_dt = []
        first = 0
        for i in range(len(inp_columns2)):
            if first == 0:
                if str(inp_columns2[i]) != 'nan':
                    first = i
                    break

        j = first
        k = first + 1
        for i in range(first, len(inp_columns) - 1):
            if str(inp_columns[j]) != 'nan' and str(inp_columns[k]) != 'nan':
                if k - j == 1:
                    exp_change_dt.append(inp_columns[k] / inp_columns[j])
                    j += 1
                    k += 1
                elif k - j > 1:
                    exp_change_dt.append((inp_columns[k] / inp_columns[j]) ** (1 / (k - j)))
                    j = k
                    k += 1
            elif str(inp_columns[j]) != 'nan' and str(inp_columns[k]) == 'nan':
                k += 1
        return numpy.array(exp_change_dt)
    except Exception as e:
        print(e.__traceback__.tb_lineno, e.__str__())
