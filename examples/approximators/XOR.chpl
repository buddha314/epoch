



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

config const epochs: int = 100000;
config const reportInterval: int = 1000;
config const learningRate: real = 0.01;


 proc main() {
   writeln("");
   writeln("");
   writeln("XOR... starting...");
   writeln("");

   var t: Timer;
   t.start();



   var X = [[0.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 0.0, 1.0]];

   var Y = [0.0, 1.0, 1.0, 0.0];

   var dims = [X.shape[1],2,1],  // 2d inputs, 2-unit layer, 1d output
       activations = ["tanh","linear"];

   var testX = [[0.0, 0.0, 1.0, 1.0],
                [1.0, 0.0, 1.0, 0.0]];

   var testY = [1.0, 0.0, 0.0, 1.0];


   var model = new FCNetwork(dims,activations);

   model.train(X,Y,epochs,learningRate,reportInterval);

   writeln("\n\n");

   var preds = model.forwardPass(testData);
   writeln("XOR Predictions: ",preds);
   writeln("");

   t.stop();
   writeln("Training took: ",t.elapsed()," seconds");


   writeln("");
   writeln("XOR... done...");
   writeln("");
   writeln("");
 }


}
