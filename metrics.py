import numpy as np

# def masked_mae(y_pred, y_true):
#     assert len(y_pred) == len(y_true)
#     mask = y_true != 0
#     return np.fabs((y_true[mask] - y_pred[mask])).mean()

# def masked_mape(y_pred, y_true):
#     assert len(y_pred) == len(y_true)
#     mask = (np.fabs(y_true) > 0.5)
#     return np.fabs((y_true[mask] - y_pred[mask]) / y_true[mask]).mean()

# def masked_rmse(y_pred, y_true):
#     assert len(y_pred) == len(y_true)
#     mask = y_true != 0
#     return np.sqrt(np.square(y_true[mask] - y_pred[mask]).mean())

def masked_mape(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(y_pred, y_true).astype('float32'),
                      y_true))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100