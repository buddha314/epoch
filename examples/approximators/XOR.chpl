



/*
This examples trains a small neural network with one two-tanh hidden layer
    and a linear output unit.
*/






module XOR {

  use NumSuch,
      Norm,
      LinearAlgebra,
      Viterbi,
      Epoch,
      Math,
      Core,
      Neural,
      Charcoal;

config const numEpochs: int = 100000;
config const reportInterval: int = 1000;
config const learningRate: real = 0.01;
config const momentum: real = 0;
config const alphaR: real = 0;


 proc main() {
   writeln("");
   writeln("");
   writeln("XOR... starting...");
   writeln("");

   var t: Timer;
   t.start();



   var X = Matrix( [0.0, 0.0, 1.0, 1.0],
                   [0.0, 1.0, 0.0, 1.0] );

   var Y = Matrix( [0.0, 1.0, 1.0, 0.0] );

   var dims = [X.shape[1],2,1],  // 2d inputs, 2-unit layer, 1d output
       activations = ["tanh","linear"];

   var testX = Matrix( [0.0],
                       [1.0] );

   var testY = Matrix( [1.0] );


   var model = new FCNetwork(dims,activations);

   model.train(X = X
              ,Y = Y
              ,momentum = momentum
              ,epochs = numEpochs
              ,learningRate = learningRate
              ,reportInterval = reportInterval
              ,regularization = "L2"
              ,alpha = alphaR
               );

   writeln("\n\n");

   var trainingPreds = model.forwardPass(X);
   var testPreds = model.forwardPass(testX);
   writeln("XOR Training Predictions: ",trainingPreds);
   writeln("Actual Training Values: ",Y,"\n");
   writeln("XOR Test Predictions: ",testPreds);
   writeln("Actual Test Values:   ",testY);
   writeln("");

   t.stop();
   writeln("Training took: ",t.elapsed()," seconds");


   writeln("");
   writeln("XOR... done...");
   writeln("");
   writeln("");
 }


}
