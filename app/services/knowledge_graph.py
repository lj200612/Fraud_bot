from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

class KnowledgeGraph:
    def __init__(self):
        load_dotenv()
        
        # 连接Neo4j数据库
        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            auth=(
                os.getenv("NEO4J_USER", "neo4j"),
                os.getenv("NEO4J_PASSWORD", "password")
            )
        )
        
        # 初始化知识图谱
        self._init_graph()

    def _init_graph(self):
        """初始化知识图谱结构"""
        with self.driver.session() as session:
            # 创建约束
            session.run("""
                CREATE CONSTRAINT fraud_type IF NOT EXISTS
                FOR (f:FraudType) REQUIRE f.name IS UNIQUE
            """)
            
            # 创建示例数据
            session.run("""
                MERGE (f1:FraudType {name: '电信诈骗'})
                MERGE (f2:FraudType {name: '网络诈骗'})
                MERGE (f3:FraudType {name: '金融诈骗'})
                
                MERGE (m1:Method {name: '冒充公检法'})
                MERGE (m2:Method {name: '虚假投资'})
                MERGE (m3:Method {name: '网络购物'})
                
                MERGE (s1:Solution {name: '立即报警'})
                MERGE (s2:Solution {name: '保留证据'})
                MERGE (s3:Solution {name: '联系银行'})
                
                MERGE (f1)-[:USES]->(m1)
                MERGE (f1)-[:HAS_SOLUTION]->(s1)
                MERGE (f1)-[:HAS_SOLUTION]->(s2)
                
                MERGE (f2)-[:USES]->(m2)
                MERGE (f2)-[:USES]->(m3)
                MERGE (f2)-[:HAS_SOLUTION]->(s2)
                MERGE (f2)-[:HAS_SOLUTION]->(s3)
                
                MERGE (f3)-[:USES]->(m2)
                MERGE (f3)-[:HAS_SOLUTION]->(s3)
            """)

    def query(self, fraud_type: str) -> dict:
        """查询特定诈骗类型的知识"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (f:FraudType {name: $fraud_type})
                OPTIONAL MATCH (f)-[:USES]->(m:Method)
                OPTIONAL MATCH (f)-[:HAS_SOLUTION]->(s:Solution)
                RETURN f.name as fraud_type,
                       collect(distinct m.name) as methods,
                       collect(distinct s.name) as solutions
            """, fraud_type=fraud_type)
            
            record = result.single()
            if record:
                return {
                    "fraud_type": record["fraud_type"],
                    "methods": record["methods"],
                    "solutions": record["solutions"]
                }
            return None

    def add_knowledge(self, fraud_type: str, methods: list, solutions: list):
        """添加新的诈骗知识"""
        with self.driver.session() as session:
            session.run("""
                MERGE (f:FraudType {name: $fraud_type})
                WITH f
                UNWIND $methods as method
                MERGE (m:Method {name: method})
                MERGE (f)-[:USES]->(m)
                WITH f
                UNWIND $solutions as solution
                MERGE (s:Solution {name: solution})
                MERGE (f)-[:HAS_SOLUTION]->(s)
            """, fraud_type=fraud_type, methods=methods, solutions=solutions)

    def close(self):
        """关闭数据库连接"""
        self.driver.close() 