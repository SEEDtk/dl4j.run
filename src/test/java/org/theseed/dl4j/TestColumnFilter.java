/**
 *
 */
package org.theseed.dl4j;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.theseed.test.Matchers.*;

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
            assertThat(colName, filter.allows(colName), isTrue());
        List<String> fieldNames = Arrays.asList("z0", "a1", "b2", "c3", "d4", "e5");
        filter = new SubsetColumnFilter(fieldNames, 7, metaCols, labelCols);
        for (String colName : colNames)
            assertThat(colName, filter.allows(colName), isTrue());
        filter = new SubsetColumnFilter(fieldNames, 3, metaCols, labelCols);
        assertThat(filter.allows("a1"), isTrue());
        assertThat(filter.allows("b2"), isTrue());
        assertThat(filter.allows("xx"), isTrue());
        assertThat(filter.allows("yy"), isTrue());
        assertThat(filter.allows("zz"), isTrue());
        assertThat(filter.allows("c3"), isFalse());
        assertThat(filter.allows("d4"), isFalse());
        assertThat(filter.allows("e5"), isFalse());
    }

}
