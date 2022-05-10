# ML-hierarchy
Custom Loss function for integrating hierarchical information into model.

Built a custom loss function with cascading loss over hierarchy tree, taking the custom loss as the tree distance between try node and predicted node.

Run with a base architecture of ResNet-9, cross-entropy per layer loss and argmax as the cascading predictor (only for leaves). 

                Train Loss : Validation Loss : Validation Accuracy : Tree Loss
    Base-ResNet9  0.2447         1.0960            0.7287.           1.8178
    Hrch-ResNet9  0.2756         2.4875            0.7400            1.7394
