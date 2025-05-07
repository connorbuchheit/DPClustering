from .kmeans import KMeans

fit = KMeans.fit
predict = KMeans.predict
balcanetal_fit = KMeans.balcanetal_fit

__all__ = [
    "fit",
    "balcanetal_fit",
    "predict",
]


