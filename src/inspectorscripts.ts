export
namespace Languages {
    export
        type LanguageModel = {
            initScript: string;
            queryCommand: string;
            matrixQueryCommand: string;
        }
}

export
    abstract class Languages {
    /**
     * Init and query script for supported languages.
     */

    static py_script: string = `import json
from sys import getsizeof
from IPython import get_ipython
from IPython.core.magics.namespace import NamespaceMagics


_jupyterlab_variableinspector_nms = NamespaceMagics()
_jupyterlab_variableinspector_Jupyter = get_ipython()
_jupyterlab_variableinspector_nms.shell = _jupyterlab_variableinspector_Jupyter.kernel.shell

try:
    import numpy as np
except ImportError:
    np = None

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import pyspark
except ImportError:
    pyspark = None

try:
    import tensorflow as tf
    import keras.backend as K
except ImportError:
    tf = None

try:
    import torch
except ImportError:
    torch = None

def _jupyterlab_variableinspector_gettypeof(x):
  if tf and isinstance(x, tf.Variable):
      return "Tensorflow Tensor"
  if tf and isinstance(x, tf.Tensor):
      return "Tensorflow Tensor"
  if torch and isinstance(x, torch.Tensor):
      if x.is_cuda:
        return 'Torch Tensor [Device {}: {}]'.format(x.get_device(), torch.cuda.get_device_name(x.get_device())))
      else:
        return 'Torch Tensor [Device: CPU]'
  if str(type(x).__name__)[0].isupper() == True:
      return str(type(x).__name__)
  if str(type(x).__name__)[0].isupper() == False:
      return str(type(x).__name__).capitalize()
  return None

def _jupyterlab_variableinspector_getsizeof(x):
    if type(x).__name__ in ['ndarray', 'Series']:
        return x.nbytes
    elif pyspark and isinstance(x, pyspark.sql.DataFrame):
        return "?"
    elif tf and isinstance(x, tf.Variable):
        return "?"
    elif pd and type(x).__name__ == 'DataFrame':
        return x.memory_usage().sum()
    else:
        return getsizeof(x)


def _jupyterlab_variableinspector_getshapeof(x):
    if isinstance(x, list):
        return "[%s]" % len(x)
    if pd and isinstance(x, pd.DataFrame):
        return "[%d x %d]" % x.shape
    if pd and isinstance(x, pd.Series):
        return "[%d]" % x.shape
    if np and isinstance(x, np.ndarray):
        shape = " x ".join([str(i) for i in x.shape])
        return "[%s]" %  shape
    if pyspark and isinstance(x, pyspark.sql.DataFrame):
        return "[? x %d]" % len(x.columns)
    if tf and isinstance(x, tf.Variable):
        shape = " x ".join([str(int(i)) for i in x.shape])
        return "[%s]" % shape
    if tf and isinstance(x, tf.Tensor):
        shape = " x ".join([str(int(i)) for i in x.shape])
        return "[%s]" % shape
    if torch and isinstance(x, torch.Tensor):
        shape = " x ".join([str(int(i)) for i in x.shape])
        return "[%s]" % shape
    return None


def _jupyterlab_variableinspector_getcontentof(x):
    # returns content in a friendly way for python variables
    # pandas and numpy
    maxoutput = 100
    if isinstance(x, list):
        truncate = 1
        content = str(x[:truncate])
        while len(content) < maxoutput:
          truncate += 1
          content = str(x[:truncate])
        content = str(x[:truncate-1] + ['...'])
    if pd and isinstance(x, pd.DataFrame):
        colnames = ', '.join(x.columns.map(str))
        content = "Column Names: %s" % colnames
    elif pd and isinstance(x, pd.Series):
        content = "Series [%d Rows]" % x.shape
    elif np and isinstance(x, np.ndarray):
        content = x.__repr__()
    else:
        content = str(x)
    if len(content) > maxoutput:
        return content[:maxoutput] + " ..."
    else:
        return content


def _jupyterlab_variableinspector_is_matrix(x):
    # True if type(x).__name__ in ["DataFrame", "ndarray", "Series"] else False
    if pd and isinstance(x, pd.DataFrame):
        return True
    if pd and isinstance(x, pd.Series):
        return True
    if np and isinstance(x, np.ndarray) and len(x.shape) <= 2:
        return True
    if pyspark and isinstance(x, pyspark.sql.DataFrame):
        return True
    if tf and isinstance(x, tf.Variable) and len(x.shape) <= 2:
        return True
    if tf and isinstance(x, tf.Tensor) and len(x.shape) <= 2:
        return True
    if torch and isinstance(x, torch.Tensor) and len(x.shape) <= 2:
        return True
    return False


def _jupyterlab_variableinspector_dict_list():
    def keep_cond(v):
        try:
            obj = eval(v)
            if isinstance(obj, str):
                return True
            if tf and isinstance(obj, tf.Variable):
                return True
            if pd and pd is not None and (
                isinstance(obj, pd.core.frame.DataFrame)
                or isinstance(obj, pd.core.series.Series)):
                return True
            if str(obj)[0] == "<":
                return False
            if  v in ['np', 'pd', 'pyspark', 'tf']:
                return obj is not None
            if str(obj).startswith("_Feature"):
                # removes tf/keras objects
                return False
            return True
        except:
            return False
    values = _jupyterlab_variableinspector_nms.who_ls()
    vardic = [{'varName': _v,
    'varType': str(_jupyterlab_variableinspector_gettypeof(eval(_v))),
    'varSize': str(_jupyterlab_variableinspector_getsizeof(eval(_v))),
    'varShape': str(_jupyterlab_variableinspector_getshapeof(eval(_v))) if _jupyterlab_variableinspector_getshapeof(eval(_v)) else '',
    'varContent': str(_jupyterlab_variableinspector_getcontentof(eval(_v))),
    'isMatrix': _jupyterlab_variableinspector_is_matrix(eval(_v))}
            for _v in values if keep_cond(_v)]
    return json.dumps(vardic, ensure_ascii=False)


def _jupyterlab_variableinspector_getmatrixcontent(x, max_rows=10000):

    # to do: add something to handle this in the future
    threshold = max_rows

    if pd and pyspark and isinstance(x, pyspark.sql.DataFrame):
        df = x.limit(threshold).toPandas()
        return _jupyterlab_variableinspector_getmatrixcontent(df.copy())
    elif np and pd and type(x).__name__ in ["Series", "DataFrame"]:
        if threshold is not None:
            x = x.head(threshold)
        x.columns = x.columns.map(str)
        return x.to_json(orient="table", default_handler=_jupyterlab_variableinspector_default)
    elif np and pd and type(x).__name__ in ["ndarray"]:
        df = pd.DataFrame(x)
        if threshold is not None:
            df = df.head(threshold)
        df.columns = df.columns.map(str)
        return df.to_json(orient="table", default_handler=_jupyterlab_variableinspector_default)
    elif tf and (isinstance(x, tf.Variable) or isinstance(x, tf.Tensor)):
        df = K.get_value(x)
        return _jupyterlab_variableinspector_getmatrixcontent(df)

def _jupyterlab_variableinspector_default(o):
    if isinstance(o, np.number): return int(o)
    raise TypeError
`;

    static scripts: { [index: string]: Languages.LanguageModel } = {
        "python3": {
            initScript: Languages.py_script,
            queryCommand: "_jupyterlab_variableinspector_dict_list()",
            matrixQueryCommand: "_jupyterlab_variableinspector_getmatrixcontent"
        },
        "python2": {
            initScript: Languages.py_script,
            queryCommand: "_jupyterlab_variableinspector_dict_list()",
            matrixQueryCommand: "_jupyterlab_variableinspector_getmatrixcontent"
        },
        "python": {
            initScript: Languages.py_script,
            queryCommand: "_jupyterlab_variableinspector_dict_list()",
            matrixQueryCommand: "_jupyterlab_variableinspector_getmatrixcontent"
        }
    };

    public static getScript( lang: string ): Promise<Languages.LanguageModel> {
        return new Promise( function( resolve, reject ) {
            if ( lang in Languages.scripts ) {
                resolve( Languages.scripts[lang] );
            } else {
                reject( "Language " + lang + " not supported yet!" );
            }
        } );

    }

}
