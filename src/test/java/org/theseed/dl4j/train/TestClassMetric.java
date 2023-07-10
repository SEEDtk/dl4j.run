/**
 *
 */
package org.theseed.dl4j.train;

import java.util.List;

import org.junit.jupiter.api.Test;
import org.nd4j.evaluation.classification.ConfusionMatrix;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.*;

/**
 * @author Bruce Parrello
 *
 */
public class TestClassMetric {

    private List<Integer> CLASS_LIST = List.of(0, 1);

    /**
     * Test the various class metrics.
     */
    @Test
    public void testMatrix() {
        ConfusionMatrix<Integer> matrix = new ConfusionMatrix<Integer>(CLASS_LIST);
        matrix.add(1, 1, 20);		// true positive
        matrix.add(0, 0, 1820);		// true negative
        matrix.add(0, 1, 180);		// false positive
        matrix.add(1, 0, 10);		// false negative
        assertThat(ClassMetric.ACCURACY.compute(matrix), closeTo(.9064, 0.001));
        assertThat(ClassMetric.NPV.compute(matrix), closeTo(.9945, 0.001));
        assertThat(ClassMetric.PRECISION.compute(matrix), closeTo(.1000, 0.001));
        assertThat(ClassMetric.SENSITIVITY.compute(matrix), closeTo(.6667, 0.001));
        assertThat(ClassMetric.SPECIFICITY.compute(matrix), closeTo(.9100, 0.001));
        assertThat(ClassMetric.NLR.compute(matrix), closeTo(Math.log10(7.2586), 0.001));
        assertThat(ClassMetric.PLR.compute(matrix), closeTo(Math.log10(0.8609), 0.001));
    }


}
