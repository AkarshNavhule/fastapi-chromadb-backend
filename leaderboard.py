from google.cloud import firestore
import random
from datetime import datetime
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "sahayak-d88d3-2e1f13a7b2bc.json"
# Initialize Firestore client
db = firestore.Client()

# Sample data for generating realistic student records
student_names = [
    "Arjun Sharma", "Priya Patel", "Rahul Kumar", "Sneha Reddy", 
    "Aditya Singh", "Kavya Nair", "Rohan Gupta", "Ananya Iyer",
    "Vikram Joshi", "Meera Agarwal"
]

# Subject-specific feedback templates
feedback_templates = {
    'maths': {
        'excellent': ["Outstanding problem-solving skills", "Exceptional mathematical reasoning"],
        'good': ["Good grasp of concepts", "Neat calculations", "Well-structured solutions"],
        'average': ["Need to practice more word problems", "Focus on formula memorization", "Improve calculation speed"],
        'poor': ["Requires basic concept revision", "Practice multiplication tables", "Seek extra help for geometry"]
    },
    'science': {
        'excellent': ["Excellent understanding of scientific concepts", "Great practical knowledge"],
        'good': ["Good theoretical knowledge", "Clear diagram representations", "Well-explained answers"],
        'average': ["Need to focus on practical applications", "Improve diagram labeling", "Study environmental science more"],
        'poor': ["Revise basic scientific principles", "Practice more experiments", "Focus on physics formulas"]
    },
    'social': {
        'excellent': ["Exceptional knowledge of history and geography", "Great analytical skills"],
        'good': ["Good understanding of civics", "Well-structured answers", "Good map reading skills"],
        'average': ["Need to improve essay writing", "Focus on current affairs", "Practice more map work"],
        'poor': ["Revise Indian history thoroughly", "Improve handwriting for long answers", "Study constitution basics"]
    },
    'english': {
        'excellent': ["Excellent vocabulary and grammar", "Outstanding creative writing"],
        'good': ["Good comprehension skills", "Clear expression of ideas", "Good vocabulary usage"],
        'average': ["Need to focus on grammar", "Improve handwriting", "Practice more essay writing"],
        'poor': ["Basic grammar revision needed", "Focus on spelling mistakes", "Read more stories for comprehension"]
    },
    'kannada': {
        'excellent': ["ಅತ್ಯುತ್ತಮ ಭಾಷಾ ಪ್ರಾವೀಣ್ಯತೆ", "Excellent language proficiency"],
        'good': ["Good understanding of Kannada literature", "Clear handwriting in Kannada", "Good translation skills"],
        'average': ["Need to practice Kannada writing", "Focus on grammar rules", "Improve vocabulary"],
        'poor': ["Basic Kannada grammar revision needed", "Practice more letter writing", "Focus on prose comprehension"]
    }
}

def get_grade_category(marks):
    """Determine grade category based on marks"""
    if marks >= 90:
        return 'excellent'
    elif marks >= 75:
        return 'good'
    elif marks >= 50:
        return 'average'
    else:
        return 'poor'

def generate_subject_feedback(subject, marks):
    """Generate feedback based on subject and marks"""
    category = get_grade_category(marks)
    feedbacks = feedback_templates[subject][category]
    return random.choice(feedbacks)

def generate_overall_feedback(percentage, rank):
    """Generate overall performance feedback"""
    if percentage >= 90:
        return f"Excellent performance! Ranked {rank}. Keep up the outstanding work!"
    elif percentage >= 75:
        return f"Good performance! Ranked {rank}. With little more effort, you can achieve excellence."
    elif percentage >= 50:
        return f"Average performance. Ranked {rank}. Focus on weak subjects to improve overall score."
    else:
        return f"Below average performance. Ranked {rank}. Requires immediate attention and extra coaching."

def create_student_leaderboard():
    """Create and populate student leaderboard in Firestore"""
    
    students_data = []
    
    # Generate data for 10 students
    for i, name in enumerate(student_names):
        # Generate realistic marks (some correlation between subjects)
        base_performance = random.randint(40, 95)  # Base performance level
        
        maths_marks = max(0, min(100, base_performance + random.randint(-15, 15)))
        science_marks = max(0, min(100, base_performance + random.randint(-12, 12)))
        social_marks = max(0, min(100, base_performance + random.randint(-10, 10)))
        english_marks = max(0, min(100, base_performance + random.randint(-8, 8)))
        kannada_marks = max(0, min(100, base_performance + random.randint(-10, 10)))
        
        # Calculate total and percentage
        all_subject_marks = maths_marks + science_marks + social_marks + english_marks + kannada_marks
        percentage = round(all_subject_marks / 5, 2)
        
        # Generate subject-specific feedbacks
        feedbacks = []
        feedbacks.append(generate_subject_feedback('maths', maths_marks))
        feedbacks.append(generate_subject_feedback('science', science_marks))
        feedbacks.append(generate_subject_feedback('social', social_marks))
        feedbacks.append(generate_subject_feedback('english', english_marks))
        feedbacks.append(generate_subject_feedback('kannada', kannada_marks))
        
        # Add some general feedbacks
        if maths_marks < 50:
            feedbacks.append("Mathematics needs immediate attention")
        if english_marks < 60:
            feedbacks.append("Focus on English grammar and vocabulary")
        if all([maths_marks >= 85, science_marks >= 85]):
            feedbacks.append("Strong in STEM subjects - consider science stream")
        if percentage >= 90:
            feedbacks.append("Exceptional overall performance!")
        
        student_data = {
            'student_name': name,
            'student_id': f'STU{2025}{str(i+1).zfill(3)}',  # e.g., STU2025001
            'maths_marks': maths_marks,
            'science_marks': science_marks,
            'social_marks': social_marks,
            'english_marks': english_marks,
            'kannada_marks': kannada_marks,
            'all_subject_marks': all_subject_marks,
            'percentage': percentage,
            'feedbacks': feedbacks,
            'created_at': datetime.now(),
            'exam_date': '2025-07-20',
            'class': '10th Standard',
            'section': 'A'
        }
        
        students_data.append(student_data)
    
    # Sort students by percentage for ranking
    students_data.sort(key=lambda x: x['percentage'], reverse=True)
    
    # Assign ranks and add overall feedback
    for i, student in enumerate(students_data):
        student['rank'] = i + 1
        student['feedbacks'].append(generate_overall_feedback(student['percentage'], student['rank']))
    
    # Store in Firestore
    collection_ref = db.collection('student_leaderboard')
    
    print("Adding students to Firestore...")
    for student in students_data:
        doc_ref = collection_ref.document(student['student_id'])
        doc_ref.set(student)
        print(f"Added: {student['student_name']} - Rank: {student['rank']} - Percentage: {student['percentage']}%")
    
    print(f"\nSuccessfully added {len(students_data)} students to Firestore!")
    return students_data

def display_leaderboard():
    """Retrieve and display leaderboard from Firestore"""
    try:
        collection_ref = db.collection('student_leaderboard')
        students = collection_ref.order_by('rank').stream()
        
        print("\n" + "="*80)
        print("🏆 STUDENT LEADERBOARD - 10th Standard Section A 🏆")
        print("="*80)
        print(f"{'Rank':<4} {'Student Name':<15} {'Math':<4} {'Sci':<4} {'Soc':<4} {'Eng':<4} {'Kan':<4} {'Total':<5} {'%':<6}")
        print("-"*80)
        
        for student_doc in students:
            student = student_doc.to_dict()
            print(f"{student['rank']:<4} {student['student_name']:<15} "
                  f"{student['maths_marks']:<4} {student['science_marks']:<4} "
                  f"{student['social_marks']:<4} {student['english_marks']:<4} "
                  f"{student['kannada_marks']:<4} {student['all_subject_marks']:<5} "
                  f"{student['percentage']:<6}%")
        
        print("-"*80)
        
    except Exception as e:
        print(f"Error retrieving leaderboard: {e}")

def get_student_detailed_report(student_id):
    """Get detailed report for a specific student"""
    try:
        doc_ref = db.collection('student_leaderboard').document(student_id)
        student_doc = doc_ref.get()
        
        if student_doc.exists:
            student = student_doc.to_dict()
            print(f"\n📋 DETAILED REPORT FOR {student['student_name']} 📋")
            print("="*60)
            print(f"Student ID: {student['student_id']}")
            print(f"Class: {student['class']} - Section: {student['section']}")
            print(f"Exam Date: {student['exam_date']}")
            print(f"Overall Rank: {student['rank']}/10")
            print(f"Overall Percentage: {student['percentage']}%")
            print("\nSUBJECT-WISE MARKS:")
            print(f"Mathematics: {student['maths_marks']}/100")
            print(f"Science: {student['science_marks']}/100")
            print(f"Social Studies: {student['social_marks']}/100")
            print(f"English: {student['english_marks']}/100")
            print(f"Kannada: {student['kannada_marks']}/100")
            print(f"Total: {student['all_subject_marks']}/500")
            
            print("\n💬 TEACHER'S FEEDBACK:")
            for i, feedback in enumerate(student['feedbacks'], 1):
                print(f"{i}. {feedback}")
                
        else:
            print(f"Student with ID {student_id} not found!")
            
    except Exception as e:
        print(f"Error retrieving student report: {e}")

def update_student_marks(student_id, subject, new_marks):
    """Update marks for a specific student and subject"""
    try:
        doc_ref = db.collection('student_leaderboard').document(student_id)
        student_doc = doc_ref.get()
        
        if student_doc.exists:
            student = student_doc.to_dict()
            
            # Update the specific subject marks
            if subject in ['maths_marks', 'science_marks', 'social_marks', 'english_marks', 'kannada_marks']:
                old_marks = student[subject]
                student[subject] = new_marks
                
                # Recalculate totals
                student['all_subject_marks'] = (student['maths_marks'] + student['science_marks'] + 
                                               student['social_marks'] + student['english_marks'] + 
                                               student['kannada_marks'])
                student['percentage'] = round(student['all_subject_marks'] / 5, 2)
                
                # Update feedback for the changed subject
                subject_name = subject.replace('_marks', '')
                new_feedback = generate_subject_feedback(subject_name, new_marks)
                
                # Update document
                doc_ref.update(student)
                
                print(f"Updated {student['student_name']}'s {subject}: {old_marks} → {new_marks}")
                print(f"New percentage: {student['percentage']}%")
                print(f"New feedback: {new_feedback}")
                
        else:
            print(f"Student with ID {student_id} not found!")
            
    except Exception as e:
        print(f"Error updating student marks: {e}")

# Main execution
if __name__ == "__main__":
    # Create the leaderboard
    students = create_student_leaderboard()
    
    # Display the leaderboard
    display_leaderboard()
    
    # Example: Get detailed report for first student
    if students:
        first_student_id = students[0]['student_id']
        get_student_detailed_report(first_student_id)
    
    # Example: Update marks (uncomment to test)
    # update_student_marks('STU2025001', 'maths_marks', 95)
