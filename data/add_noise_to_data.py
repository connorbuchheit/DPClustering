# THIS IS CURRENTLY ONLY FOR 2D DATA — TODO: MAKE IT WORK FOR MANY-DIMENSIONAL DATA?
import numpy as np 

def synth_data(b, z, epsilon, k):
  # Say I were to bin. Think about this! k bins, data from -b to b.
  # That's a 2b range. Want to be inclusive...2b/k so I guess we go from starting bounds
  # Check bound <= num <= bound + 2b/k for each bound from {-b, ..., b-2b/k}.
  inc = 2 * b / k # Histogram increment
  h = np.zeros((k, k)) # zeros of shape k k
  for x, y in z:
    x_ind = min(k-1, int(k * (x + b) / (2 * b))) # This should be proper bounding. Transforms data range
    y_ind = min(k-1, int(k * (y + b) / (2 * b))) # from [0, k-1], handles edge case x, y=b.
    h[x_ind, y_ind] += 1 # Increment properly

  # Now, make histogram DP. From LN
  noise = np.random.laplace(0, 1 / epsilon, size=(k,k))
  dp_h = h + noise
  dp_h = np.maximum(dp_h, 0) # for negative counts.

  # Data generation — remember to round because of non-integer counts due to noise
  z_tilde = []
  for i in range(k):
    for j in range(k):
      count = int(round(dp_h[i, j])) # Rounds to nearest number, makes integer.
      for _ in range(count):
        x_synth = -b + i * inc + np.random.uniform(0, inc) # Randomly anywhere in given interval
        y_synth = -b + j * inc + np.random.uniform(0, inc)
        z_tilde.append((x_synth, y_synth))

  return np.array(z_tilde)

