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
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def nuke(self):
        '''
        Deletes everything from the neo4j database.
        '''
        with self.driver.session(database="neo4j") as session:
            session.execute_write(self._run_trans, "MATCH (n) DETACH DELETE n")

    @staticmethod
    def _run_trans(tx, cmd):
        tx.run(cmd)
