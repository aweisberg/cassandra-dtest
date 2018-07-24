import re
import logging
import types
from struct import pack
from uuid import UUID

from cassandra import ConsistencyLevel
from cassandra.query import SimpleStatement
from ccmlib.node import Node

from dtest import Tester
from tools.misc import ImmutableMapping
from tools.jmxutils import JolokiaAgent, make_mbean
from tools.data import rows_to_list
from tools.assertions import (assert_all, assert_invalid, assert_length_equal,
                              assert_none, assert_one, assert_unavailable)

from cassandra.metadata import Murmur3Token, OrderedDict
import pytest


logging.getLogger('cassandra').setLevel(logging.CRITICAL)

NODELOCAL = 11
class SSTable(object):

    def __init__(self, name, repaired, pending_id):
        self.name = name
        self.repaired = repaired
        self.pending_id = pending_id


class TableMetrics(object):

    def __init__(self, node, keyspace, table):
        assert isinstance(node, Node)
        self.jmx = JolokiaAgent(node)
        self.write_latency_mbean = make_mbean("metrics", type="Table", name="WriteLatency", keyspace=keyspace, scope=table)
        self.speculative_reads_mbean = make_mbean("metrics", type="Table", name="SpeculativeRetries", keyspace=keyspace, scope=table)
        self.transient_writes_mbean = make_mbean("metrics", type="Table", name="TransientWrites", keyspace=keyspace, scope=table)

    @property
    def write_count(self):
        return self.jmx.read_attribute(self.write_latency_mbean, "Count")

    @property
    def speculative_reads(self):
        return self.jmx.read_attribute(self.speculative_reads_mbean, "Count")

    @property
    def transient_writes(self):
        return self.jmx.read_attribute(self.transient_writes_mbean, "Count")

    def start(self):
        self.jmx.start()

    def stop(self):
        self.jmx.stop()

    def __enter__(self):
        """ For contextmanager-style usage. """
        self.start()
        return self

    def __exit__(self, exc_type, value, traceback):
        """ For contextmanager-style usage. """
        self.stop()


class StorageProxy(object):

    def __init__(self, node):
        assert isinstance(node, Node)
        self.node = node
        self.jmx = JolokiaAgent(node)
        self.mbean = make_mbean("db", type="StorageProxy")

    def start(self):
        self.jmx.start()

    def stop(self):
        self.jmx.stop()

    @property
    def blocking_read_repair(self):
        return self.jmx.read_attribute(self.mbean, "ReadRepairRepairedBlocking")

    @property
    def speculated_data_request(self):
        return self.jmx.read_attribute(self.mbean, "ReadRepairSpeculatedRequest")

    @property
    def speculated_data_repair(self):
        return self.jmx.read_attribute(self.mbean, "ReadRepairSpeculatedRepair")

    def __enter__(self):
        """ For contextmanager-style usage. """
        self.start()
        return self

    def __exit__(self, exc_type, value, traceback):
        """ For contextmanager-style usage. """
        self.stop()

class StorageService(object):

    def __init__(self, node):
        assert isinstance(node, Node)
        self.node = node
        self.jmx = JolokiaAgent(node)
        self.mbean = make_mbean("db", type="StorageService")

    def start(self):
        self.jmx.start()

    def stop(self):
        self.jmx.stop()

    def get_replicas(self, ks, cf, key):
        return self.jmx.execute_method(self.mbean, "getNaturalEndpointsWithPort(java.lang.String,java.lang.String,java.lang.String,boolean)", [ks, cf, key, True])

    def __enter__(self):
        """ For contextmanager-style usage. """
        self.start()
        return self

    def __exit__(self, exc_type, value, traceback):
        """ For contextmanager-style usage. """
        self.stop()

def patch_start(startable):
    old_start = startable.start

    def new_start(self, *args, **kwargs):
        kwargs['jvm_args'] = kwargs.get('jvm_args', []) + ['-XX:-PerfDisableSharedMem',
                                                           '-Dcassandra.enable_nodelocal_queries=true']
        return old_start(*args, **kwargs)

    startable.start = types.MethodType(new_start, startable)
    return startable

def get_sstable_data(cls, node, keyspace):
    _sstable_name = re.compile('SSTable: (.+)')
    _repaired_at = re.compile('Repaired at: (\d+)')
    _pending_repair = re.compile('Pending repair: (\-\-|null|[a-f0-9\-]+)')

    out = node.run_sstablemetadata(keyspace=keyspace).stdout

    def matches(pattern):
        return filter(None, [pattern.match(l) for l in out.decode("utf-8").split('\n')])
    names = [m.group(1) for m in matches(_sstable_name)]
    repaired_times = [int(m.group(1)) for m in matches(_repaired_at)]

    def uuid_or_none(s):
        return None if s == 'null' or s == '--' else UUID(s)
    pending_repairs = [uuid_or_none(m.group(1)) for m in matches(_pending_repair)]
    assert names
    assert repaired_times
    assert pending_repairs
    assert len(names) == len(repaired_times) == len(pending_repairs)
    return [SSTable(*a) for a in zip(names, repaired_times, pending_repairs)]


class TransientReplicationBase(Tester):

    keyspace = "ks"
    table = "tbl"

    @pytest.fixture
    def cheap_quorums(self):
        return False

    def populate(self):
        self.cluster.populate(3, tokens=self.tokens, debug=True, install_byteman=True)

    def set_nodes(self):
        self.node1, self.node2, self.node3 = self.nodes

        # Make sure digest is not attempted against the transient node
        self.node3.byteman_submit(['./byteman/throw_on_digest.btm'])


    def replication_factor(self):
        return '3/1'

    def tokens(self):
        return [0, 1, 2]

    @pytest.fixture(scope='function', autouse=True)
    def setup_cluster(self, fixture_dtest_setup):
        self.tokens = self.tokens()

        patch_start(self.cluster)
        self.cluster.set_configuration_options(values={'hinted_handoff_enabled': False,
                                                       'num_tokens': 1,
                                                       'commitlog_sync_period_in_ms': 500,
                                                       'enable_transient_replication': True,
                                                       'dynamic_snitch': False})
        self.populate()
        self.cluster.start(wait_other_notice=True, wait_for_binary_proto=True)

        self.nodes = [patch_start(node) for node in self.cluster.nodelist()]
        self.set_nodes()

        session = self.exclusive_cql_connection(self.node3)
        replication_params = OrderedDict()
        replication_params['class'] = 'NetworkTopologyStrategy'
        replication_params['datacenter1'] = self.replication_factor()
        replication_params = ', '.join("'%s': '%s'" % (k, v) for k, v in replication_params.items())

        session.execute("CREATE KEYSPACE %s WITH REPLICATION={%s}" % (self.keyspace, replication_params))
        session.execute("CREATE TABLE %s.%s (pk int, ck int, value int, PRIMARY KEY (pk, ck))" % (self.keyspace, self.table))

    def assert_has_sstables(self, node, flush=False, compact=False):
        if flush:
            node.flush()
        if compact:
            node.nodetool(' '.join(['compact', self.keyspace, self.table]))

        sstables = node.get_sstables(self.keyspace, self.table)
        assert sstables

    def assert_has_no_sstables(self, node, flush=False, compact=False):
        if flush:
            node.flush()
        if compact:
            node.nodetool(' '.join(['compact', self.keyspace, self.table]))

        sstables = node.get_sstables(self.keyspace, self.table)
        assert not sstables

    def quorum(self, session, stmt_str):
        return session.execute(SimpleStatement(stmt_str, consistency_level=ConsistencyLevel.QUORUM))

    def nodelocal(self, session, stmt_str):
        return session.execute(SimpleStatement(stmt_str, consistency_level=NODELOCAL))

    def assert_local_rows(self, node, rows):
        assert_all(self.exclusive_cql_connection(node),
                   "SELECT * FROM %s.%s" % (self.keyspace, self.table),
                   rows,
                   cl=NODELOCAL)

    def insert_row(self, pk, ck, value, session=None, node=None):
        session = session or self.exclusive_cql_connection(node or self.node1)
        token = Murmur3Token.from_key(pack('>i', pk)).value
        assert token < self.tokens[0] or self.tokens[-1] < token   # primary replica should be node1
        self.quorum(session, "INSERT INTO %s.%s (pk, ck, value) VALUES (%s, %s, %s)" % (self.keyspace, self.table, pk, ck, value))

    def read_as_list(self, query, session=None, node=None):
        session = session or self.exclusive_cql_connection(node or self.node1)
        return rows_to_list(self.quorum(session, query))

    def table_metrics(self, node):
        return TableMetrics(node, self.keyspace, self.table)

    def format_rows(self, rows):
        fmt = "pk = %s, ck = %s, value = %s"
        rows = [fmt % tuple(row) for row in rows]
        return ",\n".join(rows)

class TestTransientReplication(TransientReplicationBase):

    def test_transient_noop_write(self):
        """ If both full replicas are available, nothing should be written to the transient replica """
        for node in self.nodes:
            self.assert_has_no_sstables(node)

        tm = lambda n: self.table_metrics(n)
        with tm(self.node1) as tm1, tm(self.node2) as tm2, tm(self.node3) as tm3:
            assert tm1.write_count == 0
            assert tm2.write_count == 0
            assert tm3.write_count == 0
            self.insert_row(1, 1, 1)
            assert tm1.write_count == 1
            assert tm2.write_count == 1
            assert tm3.write_count == 0

        self.assert_has_sstables(self.node1, flush=True)
        self.assert_has_sstables(self.node2, flush=True)
        self.assert_has_no_sstables(self.node3, flush=True)

    def test_transient_write(self):
        """ If write can't succeed on full replica, it's written to the transient node instead """
        for node in self.nodes:
            self.assert_has_no_sstables(node)

        tm = lambda n: self.table_metrics(n)
        with tm(self.node1) as tm1, tm(self.node2) as tm2, tm(self.node3) as tm3:
            self.insert_row(1, 1, 1)
            # Stop writes to the other full node
            self.node2.byteman_submit(['./byteman/stop_writes.btm'])
            self.insert_row(1, 2, 2)

        # node1 should contain both rows
        self.assert_local_rows(self.node1,
                               [[1, 1, 1],
                                [1, 2, 2]])

        # write couldn't succeed on node2, so it has only the first row
        self.assert_local_rows(self.node2,
                               [[1, 1, 1]])

        # transient replica should hold only the second row
        self.assert_local_rows(self.node3,
                               [[1, 2, 2]])

    def test_transient_full_merge_read(self):
        """ When reading, transient replica should serve a missing read """
        for node in self.nodes:
            self.assert_has_no_sstables(node)

        tm = lambda n: self.table_metrics(n)
        with tm(self.node1) as tm1, tm(self.node2) as tm2, tm(self.node3) as tm3:
            self.insert_row(1, 1, 1)
            # Stop writes to the other full node
            self.node2.byteman_submit(['./byteman/stop_writes.btm'])
            self.insert_row(1, 2, 2)

        # Stop reads from the node that will hold the second row
        self.node1.stop()

        # Whether we're reading from the full node or from the transient node, we should get consistent results
        for node in [self.node2, self.node3]:
            assert_all(self.exclusive_cql_connection(self.node2),
                       "SELECT * FROM %s.%s" % (self.keyspace, self.table),
                       [[1, 1, 1],
                        [1, 2, 2]],
                       cl=ConsistencyLevel.QUORUM)

    def test_blocking_read_repair_from_transient_node_transient_coordinator(self):
        self._test_blocking_read_repair_from_transient_node(self.node3)

    def test_blocking_read_repair_from_transient_node_full_coordinator(self):
        self._test_blocking_read_repair_from_transient_node(self.node2)

    def _test_blocking_read_repair_from_transient_node(self, read_node):
        """ When reading from transient replica, it should trigger blocking partition
        repair and send mutations to the full node  """
        for node in self.nodes:
            self.assert_has_no_sstables(node)

        self.insert_row(1, 1, 1)
        # Stop writes to the other full node
        self.node2.byteman_submit(['./byteman/stop_writes.btm'])
        self.insert_row(1, 2, 2)

        # Stop node containing all data to make sure repairs are triggered from transient node
        self.node1.stop()

        # write couldn't succeed on node2, so it has only the first row
        self.assert_local_rows(self.node2,
                               [[1, 1, 1]])

        self.assert_local_rows(self.node3,
                               [[1, 2, 2]])

        assert_all(self.exclusive_cql_connection(read_node),
                   "SELECT * FROM %s.%s where pk = 1" % (self.keyspace, self.table),
                   [[1, 1, 1],
                    [1, 2, 2]],
                   cl=ConsistencyLevel.QUORUM)

        # after the query, node should hold both rows because node3 should have triggered a blocking read repair
        self.assert_local_rows(self.node2,
                               [[1, 1, 1],
                                [1, 2, 2]])

    # TODO: failing because of
    # java.lang.UnsupportedOperationException: transient replicas are currently unsupported
    #    at org.apache.cassandra.locator.Replicas.checkFull(Replicas.java:305)
    #    at org.apache.cassandra.repair.RepairRunnable.runMayThrow(RepairRunnable.java:240)
    #    at org.apache.cassandra.utils.WrappedRunnable.run(WrappedRunnable.java:28)
    #
    # This tests repairAsync
    # def test_speculative_write_repair_cycle(self):
    #     """
    #     if one of the full replicas is not available, data should be written to the transient
    #     replica, but removed after incremental repair
    #     """
    #     for node in self.nodes:
    #         self.assert_has_no_sstables(node)

    #     self.node2.byteman_submit(['./byteman/stop_writes.btm'])
    #     # self.insert_row(1)
    #     tm = lambda n: self.table_metrics(n)
    #     with tm(self.node1) as tm1, tm(self.node2) as tm2, tm(self.node3) as tm3:
    #         assert tm1.write_count == 0
    #         assert tm2.write_count == 0
    #         assert tm3.write_count == 0
    #         self.insert_row(1, 1, 1)
    #         assert tm1.write_count == 1
    #         assert tm2.write_count == 0
    #         assert tm3.write_count == 1

    #     self.assert_has_sstables(self.node1, flush=True)
    #     self.assert_has_no_sstables(self.node2, flush=True)
    #     self.assert_has_sstables(self.node3, flush=True)

    #     self.node1.nodetool(' '.join(['repair', self.keyspace, '-pr']))

    #     self.assert_has_sstables(self.node1, compact=True)
    #     self.assert_has_sstables(self.node2, compact=True)
    #     self.assert_has_no_sstables(self.node3, compact=True)

    def test_full_to_trans_read_repair(self):
        """ Data on a full replica shouldn't be rr'd to transient replicas """
        session = self.exclusive_cql_connection(self.node1)
        for node in self.nodes:
            self.assert_has_no_sstables(node)

        self.node3.stop(wait_other_notice=True)
        self.insert_row(1, 1, 1, session=session)

        self.assert_has_sstables(self.node1, flush=True)

        self.node3.start(wait_other_notice=True)
        self.assert_has_no_sstables(self.node3, flush=True)

        self.node2.stop(wait_other_notice=True)

        assert_all(self.exclusive_cql_connection(self.node1),
                   "SELECT * FROM %s.%s WHERE pk = 1" % (self.keyspace, self.table),
                   [[1, 1, 1]],
                   cl=ConsistencyLevel.QUORUM)

        self.assert_has_sstables(self.node1, flush=True)
        self.assert_has_no_sstables(self.node3, flush=True)

        with StorageProxy(self.node1) as sp:
            assert sp.blocking_read_repair == 0

    def test_trans_to_full_read_repair(self):
        """ data on transient replicas should be rr'd to full replicas """
        session = self.exclusive_cql_connection(self.node1)
        for node in self.nodes:
            self.assert_has_no_sstables(node)

        self.node2.byteman_submit(['./byteman/stop_writes.btm'])
        self.insert_row(1, 1, 1, session=session)

        with self.table_metrics(self.node1) as tm:
            assert tm.transient_writes == 1

        self.assert_has_sstables(self.node1, flush=True)
        self.assert_has_no_sstables(self.node2, flush=True)
        self.assert_has_sstables(self.node3, flush=True)

        self.node1.stop()
        session = self.exclusive_cql_connection(self.node2)
        assert_all(session,
                   "SELECT * FROM %s.%s WHERE pk = 1" % (self.keyspace, self.table),
                   [[1, 1, 1]],
                   cl=ConsistencyLevel.QUORUM)

        with StorageProxy(self.node2) as sp:
            assert sp.blocking_read_repair == 1

        self.assert_has_sstables(self.node2, flush=True)
        self.assert_has_sstables(self.node3, flush=True)

        assert_all(session,
                   "SELECT * FROM %s.%s WHERE pk = 1" % (self.keyspace, self.table),
                   [[1, 1, 1]],
                   cl=ConsistencyLevel.QUORUM)

        # Second read should not trigger repair
        with StorageProxy(self.node2) as sp:
            assert sp.blocking_read_repair == 1

    def test_trans_to_full_read_repair_multi_row_full_coordinator(self):
        """
        Data on transient replicas should be rr'd to full replicas,
        if request was initiated by the full node.
        """
        self.__test_trans_to_full_read_repair_multi_row(self.node1)

    def test_trans_to_full_read_repair_multi_row_trans_coordinator(self):
        """
        Data on transient replicas should be rr'd to full replicas,
        if request was initiated by the transient node.
        """
        self.__test_trans_to_full_read_repair_multi_row(self.node3)

    def __test_trans_to_full_read_repair_multi_row(self, coordinator_node):
        session = self.exclusive_cql_connection(self.node1)
        for node in self.nodes:
            self.assert_has_no_sstables(node)

        self.insert_row(1, 1, 1, node=self.node1)
        self.node1.byteman_submit(['./byteman/stop_writes.btm'])

        self.insert_row(1, 2, 2, node=self.node2)

        # Fully stop, otherwise blocking read repair might not get triggered as
        # since a query
        self.node2.stop(wait_other_notice=True)

        self.node1.byteman_submit(['-u', './byteman/stop_writes.btm'])

        self.assert_local_rows(self.node1, [[1,1,1]])
        self.assert_local_rows(self.node3, [[1,2,2]])

        for i in range(1, 2):
            # Read from full node node, trigger RR
            assert_all(self.exclusive_cql_connection(coordinator_node),
                       "SELECT * FROM %s.%s WHERE pk = 1" % (self.keyspace, self.table),
                       [[1, 1, 1],
                        [1, 2, 2]],
                       cl=ConsistencyLevel.QUORUM)

            # Repeated calls do not trigger RR
            with StorageProxy(coordinator_node) as sp:
                assert sp.blocking_read_repair == 1

            # After repair
            self.assert_local_rows(self.node1, [[1,1,1],
                                                [1,2,2]])
            self.assert_local_rows(self.node3, [[1,2,2]])

    def test_cheap_quorums(self):
        """ writes shouldn't make it to transient nodes """
        session = self.exclusive_cql_connection(self.node1)
        for node in self.nodes:
            self.assert_has_no_sstables(node)

        tm = lambda n: self.table_metrics(n)

        with tm(self.node1) as tm1, tm(self.node2) as tm2, tm(self.node3) as tm3:
            assert tm1.write_count == 0
            assert tm2.write_count == 0
            assert tm3.write_count == 0
            self.insert_row(1, 1, 1, session=session)
            assert tm1.write_count == 1
            assert tm2.write_count == 1
            assert tm3.write_count == 0

    def test_speculative_write(self):
        """ if a full replica isn't responding, we should send the write to the transient replica """
        session = self.exclusive_cql_connection(self.node1)
        for node in self.nodes:
            self.assert_has_no_sstables(node)

        tm = lambda n: self.table_metrics(n)
        self.node3.nodetool('disablebinary')
        self.node3.nodetool('disablegossip')

        with tm(self.node1) as tm1, tm(self.node2) as tm2, tm(self.node3) as tm3:
            assert tm1.write_count == 0
            assert tm2.write_count == 0
            assert tm3.write_count == 0
            self.insert_row(1, 1, 1, session=session)
            assert tm1.write_count == 1
            assert tm2.write_count == 1
            assert tm3.write_count == 0

class TestTransientReplicationForwarding(TransientReplicationBase):

    def populate(self):
        self.cluster.populate(5, tokens=self.tokens, debug=True, install_byteman=True)

    def set_nodes(self):
        self.node1, self.node2, self.node3, self.node4, self.node5 = self.nodes

    def replication_factor(self):
        return '5/1'

    def tokens(self):
        return [0, 1, 2, 3, 4]

    def test_forwarding_repair_from_transient_node_from_full(self):
        self._test_forwarding_repair_from_transient_node(self.node1)
    def test_forwarding_repair_from_transient_node_from_transient(self):
        self._test_forwarding_repair_from_transient_node(self.node5)

    def _test_forwarding_repair_from_transient_node(self, read_node):
        """ When the full nodes return same digest, but transient node
        returns data, it should forwarded to both full nodes """

        # Shut down first two nodes
        self.node1.stop()
        self.node2.stop()
        # Make a write for other two full ones
        self.insert_row(1, 1, 1, node=self.node3)
        self.node1.start(wait_other_notice=True, wait_for_binary_proto=True)
        self.node2.start(wait_other_notice=True, wait_for_binary_proto=True)

        for node in [self.node1, self.node2]:
            self.assert_local_rows(node, [])

        for node in [self.node3, self.node4, self.node5]:
            self.assert_local_rows(node, [[1, 1, 1]])

        self.node3.stop()
        self.node4.stop()

        # Read from the nodes that do not have this data
        assert_all(self.exclusive_cql_connection(read_node),
                   "SELECT * FROM %s.%s WHERE pk = 1" % (self.keyspace, self.table),
                   [[1, 1, 1]],
                   cl=ConsistencyLevel.QUORUM)

        for node in [self.node1, self.node2, self.node5]:
            self.assert_local_rows(node, [[1, 1, 1]])
