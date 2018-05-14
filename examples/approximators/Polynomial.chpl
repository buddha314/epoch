










module Polynomial {

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


 proc main() {
   writeln("");
   writeln("");
   writeln("Polynomial... starting...");
   writeln("");

   var t: Timer;
   t.start();


   var dom: domain(2) = {1..1,1..1000};
   var X,Z: [dom] real;
   fillRandom(X);
   fillRandom(Z);
   X = 5*(X-0.5);
   Z = 5*(Z-0.5);
   X = Matrix(X);
   Z = Matrix(Z);
   var Y: X.type = X**2-3*X+2;
   var testX: Z.type = Z;
   var testY: Z.type = X**2-3*X+2;


   var dims = [X.shape[1],3,1],
       activations = ["tanh","linear"];


   var model = new FCNetwork(dims,activations);

   model.train(X,Y,momentum,numEpochs,learningRate,reportInterval);

   writeln("\n\n");

   var preds = model.forwardPass(testX);
   writeln("Polynomial Test Cost: ",model.loss.J(testY,preds));
   writeln("Polynomial Predictions: ",preds[1,1..6]);
   writeln("Actual Values:          ",testY[1,1..6]);
   writeln("");

   t.stop();
   writeln("Training took: ",t.elapsed()," seconds");


   writeln("");
   writeln("Polynomial... done...");
   writeln("");
   writeln("");
 }


}
