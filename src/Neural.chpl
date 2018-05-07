/*
 Another pass at the NN based on Alg. 6.3, 6.4 in Goodfellow, et. al. Chapter 6, approx p 204
 */

 module Neural {
   use LinearAlgebra,
       Time,
       Norm,
  //     Model,
       Random;


/*  A Fully Connected (FC) Neural Network is a stack of Layers  */
   class FCNetwork {
     var layerDom = {1..0},
         cacheDom = {1..0},
         layers: [layerDom] Layer,
         caches: [cacheDom] Cache, // Used if `trained` = false
         widths: [layerDom] int,
         activations: [layerDom] string,
         trained: bool = false;

     proc init(dims: [] int, activations: [] string) {
       this.layerDom = {1..dims.size - 1};
       this.cacheDom = {1..dims.size};
       var layers: [layerDom] Layer;
       var caches: [cacheDom] Cache;
       this.layers = layers;
       this.caches = caches;
       for l in layerDom {
         this.layers[l] = new Layer(activation = activations[l], udim = dims[l+1], ldim = dims[l]);
       }
     }

/*  Sends input data through a forwardPass of the Neural Network  */
     proc forwardPass(X:[]) {
       var Adom = X.domain;
       var A: [Adom] real = X;
       for l in this.layerDom {
         const Z = this.layers[l].linearForward(A);
         const A_current = this.layers[l].activationForward(Z);
         if ! this.trained {   // trained models don't need to cache anything
           this.caches[l] = new Cache();
           this.caches[l].aDom = A.domain;
           this.caches[l].A = A;
           this.caches[l].zDom = Z.domain;
           this.caches[l].Z = Z;
         }
         Adom = A_current.domain;
         A = A_current;
       }
       return A;
     }

/*  Propagate errors back through networks and cache gradients  */
     proc backwardPass(AL, Y) {
       const dAL: [AL.domain] real = -(Y/AL - ((1-Y)/(1-AL)));
       this.caches[cacheDom.size] = new Cache();
       this.caches[cacheDom.size].aDom = dAL.domain;
       this.caches[cacheDom.size].dA = dAL;
       for l in this.layerDom.low..this.layerDom.high by -1 {
         var dZ = this.layers[l].activationBackward(dA = this.caches[l+1].dA, Z = this.caches[l].Z);
         const (dW, db, dA_prev) = this.layers[l].linearBackward(dZ = dZ, this.caches[l].A);
         this.caches[l].wDom = dW.domain;
         this.caches[l].dW = dW;
         this.caches[l].bDom = db.domain;
         this.caches[l].db = db;
         this.caches[l].dA = dA_prev;
       }
     }

/* UpdateParameters using cached gradients  */
     proc updateParameters(learningRate = 0.001) {
       for l in this.layerDom {
         this.layers[l].W = this.layers[l].W - learningRate * this.caches[l].dW;
         this.layers[l].b = this.layers[l].b - learningRate * this.caches[l].db;
       }
       for l in cacheDom {
         this.caches[l] = new Cache();
       }
     }

/*  Full front and back sweep with parameter updates  */
     proc fullSweep(X:[], Y:[], learningRate:real = 0.001) {
       const output = this.forwardPass(X);
       const cost = computeCost(Y, output);
       this.backwardPass(output, Y);
       this.updateParameters(learningRate);
       return (cost, output);
     }

/*  Regular Gradient Descent Training  */
     proc train(X:[], Y:[], epochs = 100000, learningRate = 0.001, reportInterval = 1000) {
       for i in 1..epochs {
         var (cost, output) = this.fullSweep(X,Y,learningRate);
         if i % reportInterval == 0 {
           try! writeln("epoch: ",i,",  cost: ",cost,";     ",output);
         }
       }
       this.trained = true;
       const preds = this.forwardPass(X);
       const fcost = computeCost(Y, preds);
       writeln("");
       writeln("Training Done... Final Cost: ",fcost);
     }

/*  Minibatch Gradient Descent Training  */
     proc train(X:[], Y:[], epochs: int = 100000, learningRate: real = 0.001, reportInterval: int = 1000, batchsize: int) {
       var batches = 1 + X.shape[2]/batchsize: int;
       for i in 1..epochs {
         for batch in {1..batches} {
           var low: int = (batch - 1) * batchsize + 1;
           var high: int = batch * batchsize;
           if low < X.shape[2] {
             if high > X.shape[2] then high = X.shape[2];
             var (cost, output) = this.fullSweep(X[1..X.shape[1],low..high],Y[1..Y.shape[1],low..high]);
             if i % reportInterval == 0 || i == 1 {
               try! writeln("epoch: ",i,",  batch: ",batch,",  cost: ",cost,";     ",output);
             }
           }
         }
       }
       this.trained = true;
       const preds = this.forwardPass(X);
       const fcost = computeCost(Y, preds);
       writeln("");
       writeln("Training Done... Final Cost: ",fcost);
     }
   }

/*  Cache exists for the intermediate gradients and precusors
           temporarily used during traing via backprop        */
   class Cache {
     var bDom: domain(1),
         wDom: domain(2),
         aDom: domain(2),
         zDom: domain(2),
         A:[aDom] real,
         Z:[zDom] real,
         dW:[wDom] real,
         db:[bDom] real,
         dA: [aDom] real,
         dZ:[zDom] real;

     proc init() { }
   }

/*  A Layer of a Neural Network is defined by it's activation, weights, and bias  */
   class Layer {
     var wDom: domain(2),
         bDom: domain(1),
         W:[wDom] real,
         b:[bDom] real,
         g: Activation;

/*  Constructs a layer with given activation and weights/bias initialized
         with small random postive numbers                                */
     proc init(activation: string, udim: int, ldim: int, eps = 0.1) {
       this.wDom = {1..udim,1..ldim};
       this.bDom = this.wDom.dim(1);
       var W: [wDom] real;
       var b: [bDom] real;
       this.W = W;
       this.b = b;
       fillRandom(this.W);
       fillRandom(this.b);
       this.W = eps*this.W;
       this.b = eps*this.b;
       this.g = new Activation(name=activation);
     }

/*  Computes an Affine Transformation on A_prev:  Z = W.A_prev + b  */
     proc linearForward(A_prev: []) {
       const zDom: domain(2) = {this.W.domain.dim(1),A_prev.domain.dim(2)};
       var b: [zDom] real;  // this step is essentially python's "broadcasting"
       for i in zDom.dim(2) {  // this could be a `forall` i think
         b[..,i] = this.b;
       }
       const Z = this.W.dot(A_prev).plus(b);
       return Z;
     }

/*  Compute the activation of the current layer  */
     proc activationForward(Z: []) {
       const A: [Z.domain] real = this.g.f(Z);
       return A;
     }

/*  Compute the dZ precursor for backprop  */
     proc activationBackward(dA:[],Z:[]) {
       const dZ: [Z.domain] real = dA * this.g.df(Z);
       return dZ;
     }

/*  Compute the gradients dW, db, and dA_prev  */
     proc linearBackward(dZ:[], A_prev:[]) {
       const m: int = A_prev.shape[2];
       const dA_prev: [A_prev.domain] real = transpose(this.W).dot(dZ);
       const dW: [this.W.domain] real = dZ.dot(transpose(A_prev))/m;
       const db: [this.b.domain] real = rowSums(dZ)/m;
       return (dW, db, dA_prev);
     }
   }

   class Activation {
     var name: string;
     proc init(name: string) {
       this.name=name;
     }

     proc f(x: real) {
       if this.name == "relu" {
         return ramp(x);
       } else if this.name == "sigmoid" {
         return sigmoid(x);
       } else if this.name == "tanh" {
        return tanh(x);
       } else if this.name == "step" {
         return heaviside(x);
       } else if this.name == "linear" {
         return x;
       } else {
         return 0;
       }
     }

     proc df(x:real) {
       if this.name == "relu" {
         return dramp(x);
       } else if this.name == "sigmoid" {
         return dsigmoid(x);
       } else if this.name == "tanh" {
         return dtanh(x);
       } else if this.name == "step" {
         return dheaviside(x);
       } else if this.name == "linear" {
         return 1;
       } else {
         return 0;
       }
     }

     // Activation Functions
     proc ramp(x: real) {
       if x < 0 {
         return 0;
       } else {
         return x;
       }
     }

     proc sigmoid(x: real) {
       return (1/(1 + exp(-x)));
     }

     proc tanh(x: real) {
       return (exp(x) - exp(-x))/(exp(x) + exp(-x));
     }

     proc heaviside(x) {
       if x < 0 {
         return 0;
       } else {
         return 1;
       }
     }

     proc id(x) {
       return x;
     }

     // Derivates of Activation Functions
     proc dsigmoid(x) {
       return sigmoid(x) * (1 - sigmoid(x));
     }

     proc dramp(x) {
       return heaviside(x);
     }

     proc dtanh(x) {
       return 1 - (tanh(x))**2;
     }

     proc dheaviside(x) {
       if x == 0 {
         return 10000000000000000;
       } else {
         return 0;
       }
     }

     proc did(x) {
       return 1;
     }
  }

  proc computeCost(Y:[], AL:[]) {
    var Jp: [AL.domain] real = Y*log(AL) + (1-Y)*log(1-AL);
    var J: real = -(+ reduce Jp)/AL.domain.dim(2).size;
    return J;
  }

  class Loss {
    var name: string;
    proc init(name: string="DEFAULT") {
      this.name = name;
    }
    proc J(yHat: [], y:[]) {
      var r: [yHat.domain] real;
      if this.name == "DEFAULT" {
        r = yHat - y;
      } else {
        r = yHat - y;
      }
      return r;
    }
  }

}
