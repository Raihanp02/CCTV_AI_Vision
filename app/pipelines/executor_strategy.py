from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed

class Base(ABC):
    @abstractmethod
    def execute(self, pipeline, data):
        raise NotImplementedError("Subclasses must implement this method.")
    
class ThreadExecutorStrategy(Base):
    def execute(self, pipelines, data):
        with ThreadPoolExecutor(max_workers=len(pipelines)) as executor:
            futures = [
                executor.submit(pipeline.process, data)
                for pipeline in pipelines
            ]

            results = [f.result() for f in futures]
        
        return results
    
class ProcessorExecutorStrategy(Base):
    def execute(self, pipeline, data):
        pass