using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GomokuNN.Sources
{
    internal class CNNModelCache
    {
        private static CNNModelCache _instance = null;
        private static readonly object _padlock = new object();

        CNNModelCache()
        {
        }

        public static CNNModelCache Instance
        {
            get
            {
                lock (_padlock)
                {
                    if (_instance == null)
                    {
                        _instance = new CNNModelCache();
                    }
                    return _instance;
                }
            }
        }

        private Dictionary<string, InferenceSession> _models = new Dictionary<string, InferenceSession>();

        public InferenceSession LoadModel(string path)
        {
            if (!_models.ContainsKey(path)) 
            {
                _models[path] = new InferenceSession(path);
            }

            return _models[path];
        }

        public void UnloadModel(string path)
        {
            _models.Remove(path);
        }
    }
}
