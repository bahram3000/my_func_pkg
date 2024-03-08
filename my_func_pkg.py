import numpy
import numpy as np
import pandas
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from sklearn.linear_model import LinearRegression


#
def progress_iterable_condition(inp_data):
    """check the iteration condition and return True and numpy array of data"""
    try:
        assert type(inp_data) == numpy.ndarray or type(inp_data) == pandas.core.frame.DataFrame or type(
            inp_data) == pandas.core.series.Series or type(inp_data) == list or type(
            inp_data) == pandas.core.indexes.range.RangeIndex or type(inp_data) == pandas.core.indexes.base.Index
        if type(inp_data) == pandas.core.frame.DataFrame or type(inp_data) == numpy.ndarray:
            assert len(inp_data.shape) == 1
            if type(inp_data) == pandas.core.frame.DataFrame:
                inp_data2 = inp_data.values
            else:
                inp_data2 = inp_data[:]
        else:
            try:
                assert len(inp_data[0]) > 0
            except:
                if type(inp_data) == pandas.core.series.Series:
                    inp_data2 = inp_data.values
                else:
                    inp_data2 = inp_data[:]
        return True, inp_data2
    except Exception as e:
        if e.__traceback__.tb_lineno == 3:
            print('input data must a pandas Dataframe or Series or numpy array or list ')
        elif e.__traceback__.tb_lineno == 7:
            print('input data must have 1 dimension')
        elif e.__traceback__.tb_lineno == 14:
            print('your list must have 1 dimension')
        print(e.__traceback__.tb_lineno, e.__str__(), e)
        return False, False


#
def auto_correlation_function(inp_data1, inp_data2):
    """calculate correlation between to data from start to end by different walk"""
    try:
        assert type(inp_data1) == type(inp_data2)
        p1 = progress_iterable_condition(inp_data1)
        p2 = progress_iterable_condition(inp_data2)
        assert p1[0]
        assert p2[0]
        assert len(inp_data1) == len(inp_data2)
        arc = []
        for i in range(2, len(p1[1])):
            arc.append(numpy.corrcoef(p1[1][0:i], p2[1][0:i])[0][1])
        return numpy.array(arc)
    except Exception as e:
        print(e.__traceback__.tb_lineno, e.__str__(), e)


#
def cor_during_time(inp_data1, inp_data2, time_period):
    """calculate a time_period_correlation during time"""
    try:
        assert type(inp_data1) == type(inp_data2)
        p1 = progress_iterable_condition(inp_data1)
        p2 = progress_iterable_condition(inp_data2)
        assert p1[0]
        assert p2[0]
        assert len(inp_data1) == len(inp_data2)
        c_d_t = []
        for i in range(len(p1[1]) - time_period):
            c_d_t.append(numpy.corrcoef(p1[1][i:i + time_period], p2[1][i:i + time_period])[0][1])
        return numpy.array(c_d_t)
    except Exception as e:
        print(e.__traceback__.tb_lineno, e.__str__(), e)


#
def intersect(inp_data1, inp_data2):
    """intersection two sample for indexing"""
    try:
        assert type(inp_data1) == type(inp_data2)
        p1 = progress_iterable_condition(inp_data1)
        p2 = progress_iterable_condition(inp_data2)
        assert p1[0]
        sect = []
        if len(p1[1]) > len(p2[1]):
            # print(1)
            for i in p1[1]:
                if i in p2[1]:
                    sect.append(i)

        else:

            for i in p2[1]:
                if i in p1[1]:
                    sect.append(i)

        return numpy.array(sect)
    except Exception as e:
        print(e.__traceback__.tb_lineno, e.__str__(), e)


#

def exp_change(inp_columns):
    """calculate geometric average or change back to back and return array of series change"""
    try:
        p1 = progress_iterable_condition(inp_columns)
        assert p1[0]
        inp_columns2 = p1[1]
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
            if str(inp_columns2[j]) != 'nan' and str(inp_columns2[k]) != 'nan':
                if k - j == 1:
                    exp_change_dt.append(inp_columns2[k] / inp_columns2[j])
                    j += 1
                    k += 1
                elif k - j > 1:
                    exp_change_dt.append((inp_columns2[k] / inp_columns2[j]) ** (1 / (k - j)))
                    j = k
                    k += 1
            elif str(inp_columns2[j]) != 'nan' and str(inp_columns2[k]) == 'nan':
                k += 1
        return numpy.array(exp_change_dt)
    except Exception as e:
        print(e.__traceback__.tb_lineno, e.__str__(), e)


#
def mean_exp(inp_column):
    """calculate mean geometric of an return series"""
    try:
        p1 = progress_iterable_condition(inp_column)
        assert p1[0]
        inp_column2 = p1[1]
        ex = 1
        for i in inp_column2:
            ex *= i
        return ex ** (1 / len(inp_column))
    except Exception as e:
        print(e.__traceback__.tb_lineno, e.__str__(), e)


#
def build_exp_smooth_trend(inp_dataframe):
    """return smooth grom exponential by static mean return"""
    assert type(inp_dataframe) == pandas.core.frame.DataFrame or type(inp_dataframe) == numpy.ndarray or type(
        inp_dataframe) == list
    if type(inp_dataframe) == pandas.core.frame.DataFrame or type(inp_dataframe) == numpy.ndarray:
        assert len(inp_dataframe.shape) == 2
        if type(inp_dataframe) == pandas.core.frame.DataFrame:
            inp_dataframe2 = inp_dataframe.values
        else:
            inp_dataframe2 = inp_dataframe[:]
    else:
        assert len(inp_dataframe[0]) > 1
        inp_dataframe2 = numpy.array(inp_dataframe)
    return_data = numpy.array([exp_change(inp_dataframe2[:, i]) for i in range(inp_dataframe2.shape[1])]).T
    mean_return_data = numpy.array([mean_exp(return_data[:, i]) for i in range(return_data.shape[1])])
    expected_data = pandas.DataFrame([])
    for i in range(inp_dataframe2.shape[1]):
        expected_data[i] = numpy.array(
            [inp_dataframe2[0, i] * (mean_return_data[i] ** j) for j in range(inp_dataframe2.shape[0])])
    return expected_data


#
def build_exp_fit(inp_dataframe):
    """fit data on exponential curve fit on input data"""
    try:
        assert type(inp_dataframe) == pandas.core.frame.DataFrame or type(inp_dataframe) == numpy.ndarray or type(
            inp_dataframe) == list
        if type(inp_dataframe) == pandas.core.frame.DataFrame or type(inp_dataframe) == numpy.ndarray:
            assert len(inp_dataframe.shape) == 2
            if type(inp_dataframe) == pandas.core.frame.DataFrame:
                inp_dataframe2 = inp_dataframe.values
            else:
                inp_dataframe2 = inp_dataframe[:]
        else:
            assert len(inp_dataframe[0]) > 1
            inp_dataframe2 = numpy.array(inp_dataframe)

        def exp_fit(x, a_, b_):
            return a_ + b_ * numpy.exp(x)

        fit_data = pandas.DataFrame([])
        for i in range(inp_dataframe2.shape[1]):
            x_data = numpy.array([i for i in range(len(inp_dataframe2))]) / len(inp_dataframe2)
            y_data = inp_dataframe2[:, i] / numpy.max(inp_dataframe2[:, i])
            cfp = curve_fit(exp_fit, x_data, y_data)
            fit_data[i] = numpy.array(
                [exp_fit(j, cfp[0][0], cfp[0][1]) * numpy.max(inp_dataframe2[:, i]) for j in x_data])
        return fit_data
    except Exception as e:
        print(e.__traceback__.tb_lineno, e.__str__(), e)


#
def residual(inp_dataframe):
    """calculate divide residual of prediction and data"""
    assert type(inp_dataframe) == pandas.core.frame.DataFrame or type(inp_dataframe) == numpy.ndarray or type(
        inp_dataframe) == list
    if type(inp_dataframe) == pandas.core.frame.DataFrame or type(inp_dataframe) == numpy.ndarray:
        assert len(inp_dataframe.shape) == 2
        if type(inp_dataframe) == pandas.core.frame.DataFrame:
            inp_dataframe2 = inp_dataframe.values
        else:
            inp_dataframe2 = inp_dataframe[:]
    else:
        assert len(inp_dataframe[0]) > 1
        inp_dataframe2 = numpy.array(inp_dataframe)

    smooth_dt = build_exp_smooth_trend(inp_dataframe2)
    exp_fit = build_exp_fit(inp_dataframe2)
    return inp_dataframe2 / smooth_dt, inp_dataframe2 / exp_fit


#
def build_exp_smooth_final_fit(inp_dataframe):
    assert type(inp_dataframe) == pandas.core.frame.DataFrame or type(inp_dataframe) == numpy.ndarray or type(
        inp_dataframe) == list
    if type(inp_dataframe) == pandas.core.frame.DataFrame or type(inp_dataframe) == numpy.ndarray:
        assert len(inp_dataframe.shape) == 2
        if type(inp_dataframe) == pandas.core.frame.DataFrame:
            inp_dataframe2 = inp_dataframe.values
        else:
            inp_dataframe2 = inp_dataframe[:]
    else:
        assert len(inp_dataframe[0]) > 1
        inp_dataframe2 = numpy.array(inp_dataframe)
    smooth_dt = build_exp_smooth_trend(inp_dataframe)
    smooth_dt = smooth_dt.values
    res = inp_dataframe2 / smooth_dt
    mean_res = [mean_exp(res[:, i]) for i in range(res.shape[1])]
    # return mean_res
    return numpy.array([smooth_dt[:, i] * mean_res[i] for i in range(res.shape[1])]).T


#
def key_points(inp_x, inp_y, point_num: int, max_iter: int):
    """find key point of data and return errors,min_error,key_points"""
    all_errors = []
    min_s = numpy.inf
    selected_point = []
    p_x = progress_iterable_condition(inp_x)
    p_y = progress_iterable_condition(inp_y)
    assert p_x[0] and p_y[0]
    x = p_x[1]
    y = p_y[1]
    for i in range(2, point_num):
        all_err = []
        for j in range(max_iter):
            choices_x1 = numpy.random.choice(x, i)
            choices_y1 = y[choices_x1]
            f_7 = interp1d(choices_x1, choices_y1)
            range_choice1 = (numpy.min(choices_x1), numpy.max(choices_x1))
            x_7 = (x / numpy.max(x)) * (range_choice1[1] - range_choice1[0]) + range_choice1[0]
            y_7 = f_7(x_7)
            mse = mean_squared_error(y, y_7)
            all_err.append(mse)
            if mse < min_s:
                min_s = mse
                selected_point = choices_x1
        all_errors.append(all_err)
    return all_errors, min_s, selected_point


#
def key_index(inp_data, point_num: int, max_iter: int):
    """return key index of data"""
    p1 = progress_iterable_condition(inp_data)
    assert p1[0]
    y1 = p1[1]
    x1 = numpy.arange(0, len(y1))
    return numpy.sort(key_points(x1, y1, point_num, max_iter)[2])


#
def auto_interpolation_fit(inp_x, inp_y):
    """build all data by interpolation on first walk sample fragment length"""
    p_x = progress_iterable_condition(inp_x)
    p_y = progress_iterable_condition(inp_y)
    assert p_x[0] and p_y[0]
    x = p_x[1]
    y = p_y[1]
    mse = numpy.zeros(shape=(len(x),))
    for i in range(2, len(x)):
        f_2 = interp1d(x[:i + 1], y[:i + 1])
        x_2 = x / (len(x) / i)
        y_2 = f_2(x_2)
        mse[i] = (mean_squared_error(y, y_2))
    return mse[2:]


#
def sample_fractal_on_all_data(inp_x, inp_y, max_len=None, optimal_iter=False):
    """find optimal range for fractal sample on all data by return errors"""
    p_x = progress_iterable_condition(inp_x)
    p_y = progress_iterable_condition(inp_y)
    assert len(p_x[1]) == len(p_y[1])
    assert p_x[0] and p_y[0]
    assert max_len < len(p_y[1])
    x = p_x[1]
    y = p_y[1]
    # all_mse = []
    select_point = []
    select_mse = []
    if max_len:
        pass
    else:
        max_len = len(y)
    for i in tqdm(range(2, max_len)):
        # print(i, end=',')
        sub_select = []
        min_err = numpy.inf
        for j in range(0, len(x) - i):
            f_4 = interp1d(x[j:j + i + 1], y[j:j + i + 1])
            x_4 = ((x / numpy.max(x)) * i) + j
            z_2 = f_4(x_4)
            y_4 = (z_2 / numpy.max(z_2)) * numpy.max(y)
            mse = mean_squared_error(y, y_4)
            # mse3.append(mse)
            if mse < min_err:
                min_err = mse
                sub_select.append(x[j:j + i + 1])

        select_point.append(sub_select[-1])
        # all_mse.append(mse3)
        select_mse.append(min_err)
        if optimal_iter and len(select_point) > 2:
            if numpy.sum(
                    intersect(select_point[-2], select_point[-1]) == select_point[-2]) \
                    / len(intersect(select_point[-2], select_point[-1]) == select_point[-2]):
                break
    return select_mse, select_point  # ,all_mse


#
def optimal_points_sample_fractal(inp_x, inp_y, max_len=None, optimal_iter=False):
    sf = sample_fractal_on_all_data(inp_x, inp_y, max_len, optimal_iter)
    sfmse = numpy.array(sf[0])
    i = numpy.where(sfmse == numpy.min(sfmse))[0][0]
    return sf[1][i]


#
def build_from_sample_all_data(inp_x, inp_y, all_x, all_y, kind=None):
    px = progress_iterable_condition(inp_x)
    py = progress_iterable_condition(inp_y)
    pxa = progress_iterable_condition(all_x)
    pya = progress_iterable_condition(all_y)
    assert px[0] and py[0] and pxa[0] and pya[0]
    x_sample = px[1]
    y_sample = py[1]
    x_all = pxa[1]
    y_all = pya[1]
    f_1 = interp1d(x_sample, y_sample)
    range_xs = (numpy.min(x_sample), numpy.max(x_sample))
    tr_x = ((x_all / numpy.max(x_all)) * (range_xs[1] - range_xs[0])) + range_xs[0]
    if kind == 'adj':
        tr_y = (f_1(tr_x) / numpy.max(f_1(tr_x))) * numpy.max(all_y)
    else:
        tr_y = f_1(tr_x)
    return tr_y


#
def build_from_sample_all_data_feature(inp_x, inp_y, all_x):
    px = progress_iterable_condition(inp_x)
    py = progress_iterable_condition(inp_y)
    pxa = progress_iterable_condition(all_x)
    assert px[0] and py[0] and pxa[0]
    x_sample = px[1]
    y_sample = py[1]
    x_all = pxa[1]
    f_1 = interp1d(x_sample, y_sample)
    range_xs = (numpy.min(x_sample), numpy.max(x_sample))
    tr_x = ((x_all / numpy.max(x_all)) * (range_xs[1] - range_xs[0])) + range_xs[0]
    return f_1(tr_x)


#
def build_from_sample_all_data_fit(inp_x, inp_y, all_x, all_y, samp_size=0.1, kind=None):
    tr_y = build_from_sample_all_data_feature(inp_x, inp_y, all_x)
    res = all_y / tr_y
    x1 = np.arange(len(res))
    y1 = res[:]
    rsf = sample_fractal_on_all_data(x1, y1, int(len(y1) * samp_size))
    z1 = build_from_sample_all_data(rsf[1][-1], y1[rsf[1][-1]], x1, y1)
    return z1 * tr_y


#
def build_dependency_data(inp_data, walk_len):
    p1 = progress_iterable_condition(inp_data)
    assert p1[0]
    dep_x = []
    dep_y = []
    for i in range(len(p1[1]) - walk_len):
        dep_x.append(p1[1][i:i + walk_len])
        dep_y.append(p1[1][i + walk_len])
    return numpy.array(dep_x), numpy.array(dep_y)


#
class Interpolate_Predective():
    def __int__(self, inp_x, inp_y):
        self.all_x = inp_x
        self.all_y = inp_y

    def fit(self, samp_x, samp_y, max_len=None):
        self.samp_x = samp_x
        self.samp_y = samp_y
        if max_len:
            max_len2 = int(max_len * len(samp_x))
        else:
            max_len2 = int(0.1 * len(samp_x))
        p = optimal_points_sample_fractal(samp_x, samp_y, max_len2)
        z = build_from_sample_all_data_feature(p, samp_y[p], samp_x)
        lr = LinearRegression()
        lr.fit(z.reshape((-1, 1)), samp_y.reshape((-1, 1)))
        self.fiter = lr
        self.points = p
        # return self.fit

    def predict_linear(self, inp_x):
        x_all = np.concatenate([self.samp_x, numpy.delete(inp_x, np.where(inp_x == self.samp_x))])
        z = build_from_sample_all_data_feature(self.samp_x, self.samp_y, x_all)
        z_lr = self.fiter.predict(z.reshape((-1, 1)))
        z_lr = z_lr.reshape((-1,))
        # z_fit = build_from_sample_all_data_fit(self.points, self.samp_y[self.points], self.samp_x, self.samp_y, 0.1)
        # z_fit = build_from_sample_all_data_feature(np.arange(len(z_fit)), z_fit, x_all)
        return z_lr[len(self.samp_x):]

    def predict_interpolate(self, inp_x):
        x_all = np.concatenate([self.samp_x, numpy.delete(inp_x, np.where(inp_x == self.samp_x))])
        z_fit = build_from_sample_all_data_fit(self.points, self.samp_y[self.points], self.samp_x, self.samp_y, 0.1)
        z_fit = build_from_sample_all_data_feature(np.arange(len(z_fit)), z_fit, x_all)
        return z_fit[len(self.samp_x):]

    def predict_final(self, inp_x):
        return (self.predict_linear(inp_x) + self.predict_interpolate(inp_x)) / 2
