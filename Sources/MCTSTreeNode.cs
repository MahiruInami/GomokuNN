using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GomokuNN.Sources
{
    internal class MCTSTreeNode
    {
        public int PlayoutsCount { get; set; }
        public float PlayoutScore { get; set; }
        public float WinProbability { get; set; }
        public float PolicyProbability { get; set; }

        public bool IsGameEndPosition { get; set; }
        public bool IsEndPoint { get { return Leafs.Count == 0; } }

        public Constants.MovePosition MovePosition { get; set; }
        public int MoveColor { get; set; }

        public MCTSTreeNode Parent { get; set; }

        public List<MCTSTreeNode> Leafs;

        public MCTSTreeNode(MCTSTreeNode parent)
        {
            this.Parent = parent;
            Leafs = new List<MCTSTreeNode>();
        }
    }
}
