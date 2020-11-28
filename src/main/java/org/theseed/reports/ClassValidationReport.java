/**
 *
 */
package org.theseed.reports;

import java.io.OutputStream;
import java.util.List;

import org.apache.commons.lang3.StringUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.theseed.dl4j.train.ClassPredictError;

/**
 * This class produces the validation report for a classification model.
 *
 * @author Bruce Parrello
 *
 */
public class ClassValidationReport extends BaseValidationReport {

    /** list of possible label values */
    private List<String> labelValues;

    /**
     * Construct the reporting object.
     *
     * @param output	output stream
     */
    public ClassValidationReport(OutputStream output) {
        super(output);
    }

    @Override
    public void startReport(List<String> metaCols, List<String> labels) {
        this.labelValues = labels;
        // output the heading line.
        this.println(StringUtils.join(metaCols,  '\t') + "\texpected\tpredicted\terror");
    }

    @Override
    public void reportOutput(List<String> metaData, INDArray expected, INDArray output) {
        // Loop through the rows.
        for (int r = 0; r < metaData.size(); r++) {
            String rowName = metaData.get(r);
            int e = ClassPredictError.computeBest(expected, r);
            int o = ClassPredictError.computeBest(output, r);
            String error = (e == o ? "" : "X");
            this.println(StringUtils.joinWith("\t", rowName, this.labelValues.get(e), this.labelValues.get(o), error));
        }
    }

}
