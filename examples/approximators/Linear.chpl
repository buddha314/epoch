










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


 proc main() {
   writeln("");
   writeln("");
   writeln("Linear... starting...");
   writeln("");

   var t: Timer;
   t.start();


   var dom: domain(2) = {1..1,1..100};
   var X,Z: [dom] real;
   fillRandom(X);
   fillRandom(Z);
   X = 4*X;
   Z = 1000*Z;
   X = Matrix(X);
   Z = Matrix(Z);
   var Y: X.type = 0.5*X+2;
   var testX: Z.type = Z;
   var testY: Z.type = 0.5*testX+2;


   var dims = [X.shape[1],1],
       activations = ["linear"];


   var model = new FCNetwork(dims,activations);

   model.train(X,Y,numEpochs,learningRate,reportInterval);

   writeln("\n\n");

   var preds = model.forwardPass(testX);
   writeln("Line Predictions: ",preds[1,1..6]);
   writeln("Actual Values:    ",testY[1,1..6]);
   writeln("");
   writeln("The equation of the line is y=",model.layers[1].W,"x+",model.layers[1].b);

   t.stop();
   writeln("Training took: ",t.elapsed()," seconds");


   writeln("");
   writeln("Linear... done...");
   writeln("");
   writeln("");
 }


}
