










module Sine {

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
   writeln("Sine... starting...");
   writeln("");

   var t: Timer;
   t.start();


   var dom: domain(2) = {1..1,1..100};
   var X,Z: [dom] real;
   fillRandom(X);
   fillRandom(Z);
   X = 2*pi*X;
   Z = 2*pi*Z;
   X = Matrix(X);
   Z = Matrix(Z);
   var Y: X.type = sin(X);
   var testX: Z.type = Z;
   var testY: Z.type = sin(testX);


   var dims = [X.shape[1],3,1],
       activations = ["tanh","linear"];


   var model = new FCNetwork(dims,activations);

   model.train(X,Y,numEpochs,learningRate,reportInterval);

   writeln("\n\n");

   var preds = model.forwardPass(testX);
   writeln("Sine Test Cost: ",model.loss.J(testY,preds));
   writeln("Sine Predictions: ",preds[1,1..6]);
   writeln("Actual Values:    ",testY[1,1..6]);
   writeln("");

   t.stop();
   writeln("Training took: ",t.elapsed()," seconds");


   writeln("");
   writeln("Sine... done...");
   writeln("");
   writeln("");
 }


}
