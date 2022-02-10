/**
 *
 */
package org.theseed.dl4j;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.*;

import java.util.Arrays;
import java.util.List;

import org.junit.jupiter.api.Test;

/**
 * @author Bruce Parrello
 *
 */
public class TestColumnFilter {

    @Test
    public void test() {
        List<String> colNames = Arrays.asList("a1", "b2", "c3", "d4", "e5", "xx", "yy", "zz");
        List<String> metaCols = Arrays.asList("xx", "yy");
        List<String> labelCols = Arrays.asList("zz");
        BalanceColumnFilter filter = new BalanceColumnFilter.All();
        for (String colName : colNames)
            assertThat(colName, filter.allows(colName), equalTo(true));
        List<String> fieldNames = Arrays.asList("z0", "a1", "b2", "c3", "d4", "e5");
        filter = new SubsetColumnFilter(fieldNames, 7, metaCols, labelCols);
        for (String colName : colNames)
            assertThat(colName, filter.allows(colName), equalTo(true));
        filter = new SubsetColumnFilter(fieldNames, 3, metaCols, labelCols);
        assertThat(filter.allows("a1"), equalTo(true));
        assertThat(filter.allows("b2"), equalTo(true));
        assertThat(filter.allows("xx"), equalTo(true));
        assertThat(filter.allows("yy"), equalTo(true));
        assertThat(filter.allows("zz"), equalTo(true));
        assertThat(filter.allows("c3"), equalTo(false));
        assertThat(filter.allows("d4"), equalTo(false));
        assertThat(filter.allows("e5"), equalTo(false));
    }

}
