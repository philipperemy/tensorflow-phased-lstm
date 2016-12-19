# Phased LSTM: Accelerating Recurrent Network Training for Long or Event-based Sequences (NIPS 2016)

Work in progress (will be fully available in a few weeks). Please star it to see the evolution!


<div align="center">
  <img src="https://www.tensorflow.org/images/tf_logo_transp.png" width="200"><br><br>
</div>
-----------------

# Phased LSTM

The Phased LSTM model extends the LSTM model by adding a new time gate, kt (Fig. 1(b)). The
opening and closing of this gate is controlled by an independent rhythmic oscillation specified by
three parameters; updates to the cell state ct and ht are permitted only when the gate is open. The
first parameter, τ , controls the real-time period of the oscillation. The second, ron, controls the ratio
of the duration of the “open” phase to the full period. The third, s, controls the phase shift of the
oscillation to each Phased LSTM cell.

<div align="center">
  <img src="fig/fig1.png" width="400"><br><br>
</div>

<div align="center">
  <img src="fig/fig2.png" width="400"><br><br>
</div>

<div align="center">
  <img src="fig/fig3.png" width="400"><br><br>
</div>


## Resuts on MNIST data set

<div align="center">
  <img src="fig/mnist_acc.png" width="400"><br><br>
</div>

<div align="center">
  <img src="fig/mnist_ce.png" width="400"><br><br>
</div>
