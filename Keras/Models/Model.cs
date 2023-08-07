namespace Keras.Models
{
    using global::Keras.Layers;
    using Python.Runtime;
    using System.Collections.Generic;
    using static System.Net.Mime.MediaTypeNames;
    using System.IO;

    /// <summary>
    /// In the functional API, given some input tensor(s) and output tensor(s).
    /// This model will include all layers required in the computation of b given a.
    /// </summary>
    /// <seealso cref="Keras.Models.BaseModel" />
    public class Model : BaseModel
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Model"/> class.
        /// </summary>
        internal Model()
        {

        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Model" /> class.
        /// </summary>
        /// <param name="obj">The object.</param>
        internal Model(PyObject obj)
        {
            PyInstance = obj;
            Init();
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Model"/> class.
        /// </summary>
        /// <param name="inputs">The inputs layers.</param>
        /// <param name="outputs">The outputs layers.</param>
        /// <param name="name">The layer name.</param>
        public Model(BaseLayer[] inputs, BaseLayer[] outputs, string name = null)
        {
            //PyObject[] inputList = new List<PyObject>();
            //PyObject[] outputList = new List<PyObject>();

            //foreach (var item in inputs)
            //{
            //    inputList.Add(item.PyInstance);
            //}

            //foreach (var item in outputs)
            //{
            //    outputList.Add(item.PyInstance);
            //}

            this["inputs"] = inputs;
            this["outputs"] = outputs;
            this["name"] = name;

            PyInstance = Instance.keras.Model;
            Init();
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Model"/> class.
        /// </summary>
        /// <param name="inputs">The inputs layers.</param>
        /// <param name="outputs">The outputs layers.</param>
        public Model(params BaseLayer[] inputs)
        {
            List<PyObject> inputList = new List<PyObject>();
            foreach (var item in inputs)
            {
                inputList.Add(item.PyInstance);
            }


            PyInstance = Instance.keras.Model(inputList);
        }
    }
}
