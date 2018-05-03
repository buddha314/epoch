use NumSuch,
    Norm,
    LinearAlgebra,
    Viterbi,
    Epoch,
    Core,
    Neural,
    Charcoal;


class EpochTest : UnitTest {
  var nv: int = 8,
      D: domain(2) = {1..nv, 1..nv},
      SD: sparse subdomain(D),
      X: [SD] real;


  var vn: [1..0] string;

  proc setUp() {
    vn.push_back("star lord");
    vn.push_back("gamora");
    vn.push_back("groot");
    vn.push_back("drax");
    vn.push_back("rocket");
    vn.push_back("mantis");
    vn.push_back("yondu");
    vn.push_back("nebula");

    SD += (1,2); X[1,2] = 1;
    SD += (1,3); X[1,3] = 1;
    SD += (1,4); X[1,4] = 1;
    SD += (2,2); X[2,2] = 1;
    SD += (2,4); X[2,4] = 1;
    SD += (3,4); X[3,4] = 1;
    SD += (4,5); X[4,5] = 1;
    SD += (5,6); X[5,6] = 1;
    SD += (6,7); X[6,7] = 1;
    SD += (6,8); X[6,8] = 1;
    SD += (7,8); X[7,8] = 1;
  }


  proc init(verbose:bool) {
    super.init(verbose=verbose);
    this.complete();
  }

  proc testBreath(){
    writeln("");
    writeln("");
    writeln("Breathe Deep...");
    writeln("");
    writeln("");
    writeln("");
  }

  proc testHiddenLayer() {
    writeln("testHiddenLayer... starting...");
    writeln("");

    var udim = 3,
        ldim = 4,
        activation = "relu";

    var layer = new Layer(activation, udim, ldim);


    writeln("Initialized Weight Matrix: ");
    writeln(layer.W);
    writeln("Shape of W: ",layer.W.shape);
    writeln("");
    writeln("Initialized Bias Vector: ");
    writeln(layer.b);
    writeln("Shape of b: ",layer.b.shape);

    var reluP = layer.g.f(3);
    var reluN = layer.g.f(-3);

    writeln("");
    writeln("Checking Activation: relu(3)= ",reluP," and relu(-3)= ",reluN);

    assertRealApproximates("RELU Activation is Correct",expected=3,actual=reluP+reluN);

    writeln("");
    writeln("testHiddenLayer... done...");
    writeln("");
    writeln("");
  }

  proc testLinearForward() {
    writeln("");
    writeln("testLinearForward... starting...");
    writeln("");

    var X = Matrix( [1.0,0.0,0.0,0.0], [1.0,0.0,0.0,0.0], [1.0,0.0,0.0,0.0] );
    var udim = 2,
        ldim = X.shape[1],
        activation = "relu";

    var layer = new Layer(activation, udim, ldim);

    var Z = layer.linearForward(X);
    writeln(Z);
    writeln("Shapes: Z.shape = ",Z.shape,", W.shape = ",layer.W.shape,", X.shape = ",X.shape);

    assertIntEquals("Top Dim is W.shape[1]",expected=layer.W.shape[1],actual=Z.shape[1]);
    assertIntEquals("Bottom Dim is X.shape[2]",expected=X.shape[2],actual=Z.shape[2]);

    writeln("");
    writeln("testLinearForward... done...");
    writeln("");
    writeln("");
  }

  proc testActivationForward() {
    writeln("");
    writeln("testActivationForward... starting...");
    writeln("");

    var X = Matrix( [100.0,0.0,10.0,0.0], [100.0,0.0,10.0,0.0], [100.0,0.0,10.0,0.0] );
    var udim = 2,
        ldim = X.shape[1],
        activation = "relu";

    var layer = new Layer(activation, udim, ldim);

    var Z = layer.linearForward(X);
    writeln("Affine Transformation: ");
    writeln(Z);
    writeln("Shapes: Z.shape = ",Z.shape,", W.shape = ",layer.W.shape,", X.shape = ",X.shape);

    var A = layer.activationForward(Z);

    writeln("");
    writeln("This Layer's Activation: ");
    writeln(A);

    assertRealApproximates("In this test, A equals Z", expected=Z[1,1], actual=A[1,1]);

    writeln("");
    writeln("testActivationForward... done...");
    writeln("");
    writeln("");
  }

  proc testNN() {
    var layerOneUnits = 5,
        inputDim = 8,
        epochs=100000,
        batchSize = 4,
        model = new Sequential(),
        lr: real = 0.00125;

    var X = Matrix( [0,0] ,[0,1] ,[1,0] ,[1,1], [1,0], [0,0] ,[0,1] ,[1,0] ,[1,1], [1,0] ),
        y = Vector([0,1,1,0,1,0,1,1,0,1]);

    var testData = Matrix([0,1],[1,0],[1,1]);
    //model.add(new Dense(units=layerOneUnits, inputDim=inputDim, batchSize=batchSize));
    model.add(new Dense(units=2));
    //model.add(new Dense(units=2));
    model.add(new Activation(name="tanh"));
    model.fit(xTrain=X, yTrain=y, epochs=epochs, batchSize=batchSize, lr=lr);
    var predictions = model.forward_pass(testData);
    writeln("Predictions: ",predictions);
    assertIntEquals("NN correct number of layers", expected=4, actual=model.layers.size);
  }

  proc run() {
    super.run();
    testBreath();
    testHiddenLayer();
    testLinearForward();
    testActivationForward();
  //  testNN();
    return 0;
  }
}

proc main(args: [] string) : int {
  var t = new EpochTest(verbose=false);
  var ret = t.run();
  t.report();
  return ret;
}
