import sqlite3, json, networkx as nx
from rapidfuzz import fuzz
class FKGraphBuilder:
    def build(self, db_path: str, db_name: str) -> dict:
        conn = sqlite3.connect(f"{db_path}/{db_name}/{db_name}.sqlite")
        # Extract tables + columns
        tables = self._get_tables(conn)
        # Build NetworkX DiGraph
        G = nx.DiGraph()
        for table in tables:
            G.add_node(table['name'], columns=table['columns'], pk=table['pk'])
        # Add FK edges: (child_table, parent_table, {child_col, parent_col})
        for table in tables:
            for fk in self._get_foreign_keys(conn, table['name']):
                G.add_edge(table['name'], fk['parent_table'],
                child_col=fk['child_col'],
                parent_col=fk['parent_col'])
        # Compute centrality (which tables are most referenced)
        centrality = nx.in_degree_centrality(G)
        conn.close()
        # Serialize
        return {
        'nodes': [{'name': n, **G.nodes[n]} for n in G.nodes],
        'edges': [{'from': u, 'to': v, **G.edges[u,v]} for u,v in G.edges],
        'centrality': centrality
        }      
    def _get_tables(self, conn): 
        rows=conn.execute("Select name from sqlite_master where type='table'").fetchall()
        res=[]
        for row in rows:     
            table_name=row[0]
            info_rows=conn.execute(f"pragma table_info({table_name})").fetchall()
            columns= [c[1] for c in info_rows]
            pk = next((c[1] for c in info_rows if c[5] == 1), None)
            tab_dict={'name':table_name,'columns':columns ,'pk' :pk}
            res.append(tab_dict)
        return (res)
    def _get_foreign_keys(self, conn,table_name): 
        rows=conn.execute(f"PRAGMA foreign_key_list({table_name})").fetchall()
        res=[]
        for row in rows:     
            fk_dict={'parent_table':row[2],'child_col':row[3] ,'parent_col' :row[4]}
            res.append(fk_dict)
        return(res)
    