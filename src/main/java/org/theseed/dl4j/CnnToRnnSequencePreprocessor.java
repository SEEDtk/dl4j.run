/**
 *
 */
package org.theseed.dl4j;

import java.util.Arrays;

import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.util.TimeSeriesUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.api.shape.Shape;

/**
 * This class converts a convolutional representation of a sequence into a time series suitable for
 * a recurrent neural network layer.  The input has shape [batchSize, channels, 1, width] and the
 * output has shape [batchSize, channels, width].  The result is an encoding of the input sequence
 * as a progression of values over a time sequence.
 *
 * @author Bruce Parrello
 *
 */
public class CnnToRnnSequencePreprocessor implements InputPreProcessor {

    /**
     * serialization code number
     */
    private static final long serialVersionUID = 1052070337217338101L;


    @Override
    public INDArray preProcess(INDArray input, int miniBatchSize, LayerWorkspaceMgr workspaceMgr) {
        if (input.rank() != 4)
            throw new IllegalArgumentException(
                            "Invalid input: expect CNN activations with rank 4 (received input with shape "
                                            + Arrays.toString(input.shape()) + ")");
        if (input.size(2) != 1)
            throw new IllegalArgumentException(
                            "Invalid input: expect CNN activations with unit height, but height is "
                                            + input.size(2));
        //Input: 4d activations (CNN)
        //Output: 3d activations (RNN)

        if (input.ordering() != 'c' || !Shape.hasDefaultStridesForShape(input))
            input = input.dup('c');

        // Combine the height and width.
        INDArray retVal = input.reshape('c', input.size(0), input.size(1), input.size(3));
        return workspaceMgr.leverageTo(ArrayType.ACTIVATIONS, retVal);
    }

    @Override
    public INDArray backprop(INDArray output, int miniBatchSize, LayerWorkspaceMgr workspaceMgr) {
        //Output: 3d activations (RNN)
        //Input: 4d activations (CNN)

        INDArray retVal = output.reshape('c', output.size(0), output.size(1), 1, output.size(2));
        return workspaceMgr.leverageTo(ArrayType.ACTIVATION_GRAD, retVal);
    }

    @Override
    public InputPreProcessor clone() {
        return new CnnToRnnSequencePreprocessor();
    }

    @Override
    public InputType getOutputType(InputType inputType) {
        if (inputType == null || inputType.getType() != InputType.Type.CNN) {
            throw new IllegalStateException("Invalid input type: Expected input of type CNN, got " + inputType);
        }

        InputType.InputTypeConvolutional c = (InputType.InputTypeConvolutional) inputType;
        long outSize = c.getChannels() * c.getHeight() * c.getWidth();
        return InputType.recurrent(outSize);
    }

    @Override
    public Pair<INDArray, MaskState> feedForwardMaskArray(INDArray maskArray, MaskState currentMaskState,
            int minibatchSize) {
        //Assume mask array is 4d - a mask array that has been reshaped from [minibatch,timeSeriesLength] to [minibatch*timeSeriesLength, 1, 1, 1]
        if (maskArray == null) {
            return new Pair<>(maskArray, currentMaskState);
        } else {
            //Need to reshape mask array from [minibatch*timeSeriesLength, 1, 1, 1] to [minibatch,timeSeriesLength]
            return new Pair<>(TimeSeriesUtils.reshapeCnnMaskToTimeSeriesMask(maskArray, minibatchSize),currentMaskState);
        }
    }

}
