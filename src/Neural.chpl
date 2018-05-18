/*
 Another pass at the NN based on Alg. 6.3, 6.4 in Goodfellow, et. al. Chapter 6, approx p 204
 */

 module Neural {
   use LinearAlgebra,
       Time,
       Norm,
       Math,
       Model,
       Random;



/*  A Fully Connected (FC) Neural Network is a stack of Layers  */
   class FCNetwork: Model {
     var layerDom = {1..0},
         layers: [layerDom] Layer,
         widths: [1..layerDom.size+1] int,
         loss: Loss,
         activations: [layerDom] string,
         trained: bool = false;

     proc init() { }

     proc init(dims: [] int, activations: [] string) {
       this.layerDom = {1..dims.size - 1};
       var layers: [layerDom] Layer;
       this.layers = layers;
       for l in layerDom {
         this.layers[l] = new Layer(activation = activations[l], udim = dims[l+1], ldim = dims[l]);
       }
       this.widths = dims;
       this.loss = new Loss(name = activations[this.layerDom.last]);
     }

     proc inputDim() {
       return this.widths[widths.domain.first];
     }

     proc outputDim() {
       return this.widths[widths.domain.last];
     }

/*  Sends input data through a forwardPass of the Neural Network  */
     proc forwardPass(X:[]) {  // this will be predict method of the neural network
       var Adom = X.domain;
       var A: [Adom] real = X;
       for l in this.layerDom {
         const Z = this.layers[l].linearForward(A);
         const A_current = this.layers[l].activationForward(Z);
         Adom = A_current.domain;
         A = A_current;
       }
       return A;
     }

     proc forwardPass(X:[], caches: [] Cache) {
       var Adom = X.domain;
       var A: [Adom] real = X;
       for l in this.layerDom {
         const Z = this.layers[l].linearForward(A);
         const A_current = this.layers[l].activationForward(Z);
         caches[l].aDom = A.domain;
         caches[l].A = A;
         caches[l].zDom = Z.domain;
         caches[l].Z = Z;
         Adom = A_current.domain;
         A = A_current;
       }
       return A;
     }

/*  Propagate errors back through networks and cache gradients  */
     proc backwardPass(AL, Y, caches) {
       const dAL: [AL.domain] real = this.loss.dJ(Y,AL);
       caches[caches.domain.size].aDom = dAL.domain;
       caches[caches.domain.size].dA = dAL;
       for l in this.layerDom.low..this.layerDom.high by -1 {
         var dZ = this.layers[l].activationBackward(dA = caches[l+1].dA, Z = caches[l].Z);
         const (dW, db, dA_prev) = this.layers[l].linearBackward(dZ = dZ, caches[l].A);
         caches[l].wDom = dW.domain;
         caches[l].dW = dW;
         caches[l].bDom = db.domain;
         caches[l].db = db;
         caches[l].dA = dA_prev;
       }
     }

/*  UpdateParameters using cached gradients  */
     proc updateParameters(learningRate = 0.001, caches, regularizer: Regularization) {
       for l in this.layerDom {
         this.layers[l].W = this.layers[l].W - learningRate * (regularizer.Q(this.layers[l].W).plus(caches[l].dW));  // dW = V_w if momentum is being used
         this.layers[l].b = this.layers[l].b - learningRate * caches[l].db;  // db = V_b if momentum is being used
       }
       for l in caches.domain {
         delete caches[l];
         caches[l] = new Cache();
       }
     }


/*  Full front and back sweep with parameter updates  */
     proc fullSweep(X:[], Y:[], learningRate:real = 0.001) {
       var cacheDom: domain(1) = {1..this.layerDom.size + 1};
       var caches: [cacheDom] Cache;
       for l in cacheDom do caches[l] = new Cache();
       const output = this.forwardPass(X, caches);
       const cost = this.loss.J(Y, output);
       this.backwardPass(output, Y, caches);
       this.updateParameters(learningRate, caches, regularizer = new Regularization());
       delete caches;
       return (cost, output);
     }

/*  Full Sweep with Momentum and Regularization */
     proc fullSweep(X:[], Y:[], learningRate:real = 0.001, momentum: real, velCaches: [] Cache, regularizer: Regularization) {
       var cacheDom: domain(1) = {1..this.layerDom.size + 1};
       var caches: [cacheDom] Cache;
       for l in cacheDom do caches[l] = new Cache();
       const output = this.forwardPass(X, caches);
       const cost = this.loss.J(Y, output);
       this.backwardPass(output, Y, caches);
       for l in this.layerDom {
         caches[l].dW = momentum * velCaches[l].W_vel + (1 - momentum) * caches[l].dW;
         caches[l].db = momentum * velCaches[l].b_vel + (1 - momentum) * caches[l].db;
         velCaches[l].W_vel = caches[l].dW;
         velCaches[l].b_vel = caches[l].db;
       }
       this.updateParameters(learningRate, caches, regularizer);
       delete caches;
       return (cost, output);
     }

/*  Gradient Descent with Momentum and L2/L1 Regularization Options  */
     proc train(X:[],
                Y:[],
                momentum: real = 0,
                epochs = 100000,
                learningRate = 0.001,
                reportInterval = 1000,
                regularization: string = "NONE",
                alpha:real = 0) {
       var regularizer = new Regularization(regularization, alpha);
       var velCaches: [this.layerDom] Cache;
       for l in this.layerDom {
          velCaches[l] = new Cache();
          velCaches[l].wDom = this.layers[l].wDom;
          velCaches[l].bDom = this.layers[l].bDom;
       }
       for i in 1..epochs {
         var (cost, output) = this.fullSweep(X,Y,learningRate, momentum: real, velCaches, regularizer);
         if ( i % reportInterval == 0 || i == 1 ) && reportInterval < epochs {
           try! writeln("epoch: ",i,",  cost: ",cost,";     ");
         }
       }
       this.trained = true;
       const preds = this.forwardPass(X);
       const fcost = this.loss.J(Y, preds);
       writeln("");
       writeln("Training Done... Final Cost: ",fcost);
     }


/*  Gradient Descent Training with Momentum, Minibatching, and L1/L2 Regularization Options  */
     proc trainMB(X:[],
                  Y:[],
                  momentum: real = 0,
                  epochs: int = 100000,
                  learningRate: real = 0.001,
                  reportInterval: int = 1000,
                  regularization: string = "NONE",
                  alpha: real = 0,
                  batchsize: int
                 ) {
       if batchsize >= X.shape[2] {
         this.train(X, Y, momentum, epochs, learningRate, reportInterval, regularization, alpha);
       } else {
         var regularizer = new Regularization(regularization, alpha);
         var velCaches: [this.layerDom] Cache;
         for l in this.layerDom {
            velCaches[l] = new Cache();
            velCaches[l].wDom = this.layers[l].wDom;
            velCaches[l].bDom = this.layers[l].bDom;
         }
         var batches = 1 + X.shape[2]/batchsize: int;
         for i in 1..epochs {
           for batch in {1..batches} {
             var low: int = (batch - 1) * batchsize + 1;
             var high: int = batch * batchsize;
             if low < X.shape[2] {
               if high > X.shape[2] then high = X.shape[2];
               var (cost, output) = this.fullSweep(X[1..X.shape[1],low..high], Y[1..Y.shape[1],low..high],learningRate, momentum: real, velCaches, regularizer);
               if ( i % reportInterval == 0 || i == 1 ) && reportInterval < epochs {
                 try! writeln("epoch: ",i,",  batch: ",batch,"  cost: ",cost,";     ");
               }
             }
           }
         }
         this.trained = true;
         const preds = this.forwardPass(X);
         const fcost = this.loss.J(Y, preds);
         writeln("");
         writeln("Training Done... Final Cost: ",fcost);
       }
     }
   }





/*  Cache exists for the intermediate gradients and precusors
           temporarily used during traing via backprop        */
   class Cache {
     var bDom: domain(1),
         wDom: domain(2),
         aDom: domain(2),
         zDom: domain(2),
         W_vel: [wDom] real,
         b_vel: [bDom] real,
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
     proc init(activation: string, udim: int, ldim: int) {
       this.wDom = {1..udim,1..ldim};
       this.bDom = this.wDom.dim(1);
       var W: [wDom] real;
       var b: [bDom] real;
       this.W = W;
       this.b = b + 0.0000000000001;
       fillRandom(this.W);
       this.g = new Activation(name=activation);
       if this.g.name == "tanh" {
         this.W = (this.W - 0.5) * sqrt(6/(W.shape[2] + W.shape[1]));
       } else if this.g.name == "sigmoid" {
         this.W = (this.W - 0.5) * sqrt(3.6/W.shape[2]);
       } else if this.g.name == "linear" {
         this.W = (this.W - 0.5) * sqrt(1/W.shape[2]);
       } else if this.g.name == "relu" {
         this.W = this.W * sqrt(2/W.shape[2]);
       } else if this.g.name == "linear" {
         this.W = (this.W - 0.5) * sqrt(2/W.shape[2]);
       } else {
         this.W = (this.W - 0.5) * sqrt(1/W.shape[2]);
       }
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




/*  Class for Activation Function and their Derivatives  */
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
       return max(0,x);
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

/*  A class for defined various types of regularization terms  */
  class Regularization {
    var name: string,
        alpha: real;
    proc init() {
      this.name = "NONE";
      this.alpha = 0;
    }

    proc init(name: string) {
      this.name = name;
    }

    proc init(name: string, alpha: real) {
      this.name = name;
      this.alpha = alpha;
    }

    proc Q(W:[]) {
      if this.name == "L2" {
        var R: [W.domain] real = this.alpha * W;
        return R;
      } else if this.name == "L1" {
        var R: [W.domain] real = this.alpha * sgn(W);
        return R;
      } else {
        var R: [W.domain] real = 0 * W;
        return R;
      }
    }
  }


/*  Class for Assigning the Appropiate Loss Function Based on Output Layer  */
  class Loss {
    var name: string;
    proc init(name: string) {
      this.name = name;
    }
    //   Losses
    proc J(Y:[],A:[]) {
      if this.name == "sigmoid" {
        var Jp: [A.domain] real = Y*log(A) + (1-Y)*log(1-A);
        var J: real = -(+ reduce Jp)/A.domain.dim(2).size;
        return J;
      } else if this.name == "tanh" {
        var Jp: [A.domain] real = ln_2 -((1-Y)*log(1-A) + (1+Y)*log(1+A));
        var J: real = (+ reduce Jp)/(2*A.domain.dim(2).size);
        return J;
      } else if this.name == "linear" {
        var Jp: [A.domain] real = (Y - A) * (Y - A);
        var J: real = (+ reduce Jp)/(2*A.domain.dim(2).size);
        return J;
      } else {   // This catch-all-else should never be triggered
        var Jp: [A.domain] real = (Y - A)**2;
        var J: real = (+ reduce Jp)/(2*A.domain.dim(2).size);
        return J;
      }
    }

    //   Derivatives
    proc dJ(Y:[], A:[]) {
      if this.name == "sigmoid" {
        var dA: [A.domain] real = -(Y/A - ((1-Y)/(1-A)));
        return dA;
      } else if this.name == "tanh" {
        var dA: [A.domain] real = -(Y-A)/(1-A**2);
        return dA;
      } else if this.name == "linear" {
        var dA: [A.domain] real = A - Y;
        return dA;
      } else {   // This catch-all-else should never be triggered
        var dA: [A.domain] real = A - Y;
        return dA;
      }
    }
  }

}
