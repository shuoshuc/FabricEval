from neo4j import GraphDatabase

import common.flags as FLAG
from common.common import PRINTV


class GraphDB:
    '''
    GraphDB wraps the driver of backend database and translates requests from
    the Topology class.
    '''
    def __init__(self, uri, user, password):
        '''
        uri: the URI of the backend database.
        user: username
        password: password.
        '''
        # Do nothing is flag is turned off.
        self._noop = not FLAG.ENABLE_GRAPHDB
        PRINTV(1, f'Enable GraphDB is {FLAG.ENABLE_GRAPHDB}')
        if self._noop:
            return
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        if self._noop:
            return
        self.driver.close()

    def nuke(self):
        '''
        Deletes everything from the neo4j database.
        '''
        if self._noop:
            return
        with self.driver.session(database="neo4j") as session:
            session.execute_write(self._run_trans, "MATCH (n) DETACH DELETE n")

    def addCluster(self, name):
        '''
        Adds a cluster.
        '''
        if self._noop:
            return
        with self.driver.session(database="neo4j") as session:
            session.execute_write(self._run_trans,
                                  f"CREATE (:Cluster:Abstract {{name: {name}}})")

    def addAggrBlock(self, name):
        '''
        Adds an aggregation block.
        '''
        if self._noop:
            return
        with self.driver.session(database="neo4j") as session:
            session.execute_write(self._run_trans,
                                  f"CREATE (:AggrBlock:Aggr {{name: {name}}})")

    def addSwitch(self, name, stage, index, ecmp_limit):
        '''
        Adds a node.
        '''
        if self._noop:
            return
        with self.driver.session(database="neo4j") as session:
            session.execute_write(self._run_trans,
                                  f"CREATE (:Switch:Phy {{name: {name}, "
                                  f"stage: {stage}, index: {index}, "
                                  f"table: {ecmp_limit}}})")

    def addPort(self, name, index, speed, dcn_facing, host_facing):
        '''
        Adds a port.
        '''
        if self._noop:
            return
        with self.driver.session(database="neo4j") as session:
            session.execute_write(self._run_trans,
                                  f"CREATE (:Port:Phy {{name: {name}, "
                                  f"index: {index}, speed: {speed}, "
                                  f"dcn_facing: {dcn_facing}, "
                                  f"host_facing: {host_facing}}})")

    def connectAggrBlockToCluster(self, aggrblock, cluster):
        '''
        Connects an aggregation block to its parent cluster.
        '''
        if self._noop:
            return
        with self.driver.session(database="neo4j") as session:
            long_query = (f"MATCH (ab:AggrBlock {{name: {aggrblock}}}), "
                          f"(c:Cluster {{name: {cluster}}}) "
                          f"CREATE (ab)-[:MEMBER_OF]->(c)-[:PARENT_OF]->(ab)")
            session.execute_write(self._run_trans, long_query)

    def connectSwitchToAggrBlock(self, switch, aggrblock):
        '''
        Connects a switch to its parent aggregation block.
        '''
        if self._noop:
            return
        with self.driver.session(database="neo4j") as session:
            long_query = (f"MATCH (ab:AggrBlock {{name: {aggrblock}}}), "
                          f"(s:Switch {{name: {switch}}}) "
                          f"CREATE (s)-[:MEMBER_OF]->(ab)-[:PARENT_OF]->(s)")
            session.execute_write(self._run_trans, long_query)

    def connectPortToSwitch(self, port, switch):
        '''
        Connects a port to its parent switch.
        '''
        if self._noop:
            return
        with self.driver.session(database="neo4j") as session:
            long_query = (f"MATCH (p:Port {{name: {port}}}), "
                          f"(s:Switch {{name: {switch}}}) "
                          f"CREATE (p)-[:MEMBER_OF]->(s)-[:PARENT_OF]->(p)")
            session.execute_write(self._run_trans, long_query)

    def connectToRToCluster(self, switch, cluster):
        '''
        Connects a ToR (S1) switch to its parent cluster.
        '''
        if self._noop:
            return
        with self.driver.session(database="neo4j") as session:
            long_query = (f"MATCH (t:Switch {{name: {switch}}}), "
                          f"(c:Cluster {{name: {cluster}}}) "
                          f"CREATE (t)-[:MEMBER_OF]->(c)-[:PARENT_OF]->(t)")
            session.execute_write(self._run_trans, long_query)

    @staticmethod
    def _run_trans(tx, cmd):
        tx.run(cmd)
