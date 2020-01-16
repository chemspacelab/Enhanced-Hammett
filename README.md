# Enhanced-Hammett
This repository provides a python library with the implementation of the Enhanced-Hammett model [1]

It requires numpy, scipy and pandas to function, as well as matplotlib for the visualization of results

## Tutorial
This section provides a small tutorial with examples on how to use this library.

First of all import the library and the example dataset with it.

```python
import hammett as ham
from ham import experim_data
```
 
### Initial data 
 
The data should be stored in a Pandas Dataframe structure, similar to the one provided as example
 
```python
data = ham.experim_data
ham.data_show()
```
A $\rho$ will be calculated for each columns and a $\sigma$ for each row. Missing values can be handled, but they should be represented by np.NaN. There should be at least two datapoints per column in order to get the corresponding $\rho$.
The example reports kinetic constants for substituted thiols (rows) which react with different benzylbromides (columns).
 
### Find parameters

The `calc_param` function will compute a the values of $\sigma$, $\rho$ and $\k_0$.

```python
rho, sigma, k0, dicrho, dicsigma, dick0 = ham.calc_params(data)
```

### Prediction

It is possible to build a new Dataframe to compare the initial data with the prediction by calling the following function:
```python
prediction = ham.evaluate(data)
```

### Visualization
To visualize the quality of the prediction you can use the 'plot_correlation' function

```python
ham.plot_correlation(data, prediction)
```
